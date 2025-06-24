from flask import Flask, request, jsonify, render_template, Response, stream_with_context, session, send_from_directory
from flask_wtf.csrf import CSRFProtect
from rag_system import rag_system
import hashlib
import subprocess
import os
import segment.analytics as analytics
import uuid
import sys
import traceback

import requests
from html.parser import HTMLParser
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request
import json
import logging
import redis

class BodyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return ''.join(self.text)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analytics.write_key = os.getenv('SEGMENT_WRITE_KEY')

app = Flask(__name__, static_folder='templates/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = bool(os.getenv('SESSION_COOKIE_SECURE'))

csrf = CSRFProtect(app)

# Initialize Redis connection
r = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)

def validate_pow(nonce, data, difficulty):
    # Calculate the sha256 of the concatenated string of 32-bit X-Nonce header and raw body.
    # This calculation has to match the code on the client side, in index.html.
    nonce_bytes = int(nonce).to_bytes(4, byteorder='little')  # 32-bit = 4 bytes
    calculated_hash = hashlib.sha256(nonce_bytes + data).digest()
    first_uint32 = int.from_bytes(calculated_hash[:4], byteorder='big')
    return first_uint32 <= difficulty

# Shared function to generate response stream from RAG system
def generate(query, source, anonymous_id):
    full_response = ""
    try:
        for token in rag_system.answer_query_stream(query):
            yield token
            full_response += token
    except Exception as e:
        print(f"Error in RAG system: {e}", file=sys.stderr)
        traceback.print_exc()
        yield "Internal Server Error"

    if not full_response:
        full_response = "No response generated"

    if analytics.write_key:
        # Track the query and response
        analytics.track(
            anonymous_id=anonymous_id,
            event='Chatbot Question submitted',
            properties={'query': query, 'response': full_response, 'source': source}
        )
    
    return full_response

def handle_ask_request(request, session):
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # For analytics tracking, generates an anonymous id and uses it for the session
    if 'anonymous_id' not in session:
        session['anonymous_id'] = str(uuid.uuid4())
    anonymous_id = session['anonymous_id']
    
    # Determine the source based on the user agent
    user_agent = request.headers.get('User-Agent', '')
    source = 'Ask Defang Discord Bot' if 'Discord Bot' in user_agent else 'Ask Defang Website'

    # Use the shared generate function directly
    return Response(stream_with_context(generate(query, source, anonymous_id)), content_type='text/markdown')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', debug=os.getenv('DEBUG'))

@app.route('/ask', methods=['POST'])
def ask():
    if not validate_pow(request.headers.get('X-Nonce'), request.get_data(), 0x50000):
        return jsonify({"error": "Invalid Proof of Work"}), 400

    response = handle_ask_request(request, session)
    return response

# /v1/ask allows bypassing of CSRF and PoW for clients with a valid Ask Token
@app.route('/v1/ask', methods=['POST'])
@csrf.exempt
def v1_ask():
    auth_header = request.headers.get('Authorization')
    ask_token = auth_header.split("Bearer ")[1] if auth_header and auth_header.startswith("Bearer ") else None
    if ask_token and ask_token == os.getenv('ASK_TOKEN'):
        response = handle_ask_request(request, session)
        return response
    else:
        jsonify({"error": "Invalid or missing Ask Token"}), 401

@app.route('/trigger-rebuild', methods=['POST'])
@csrf.exempt
def trigger_rebuild():
    token = request.args.get('token')
    if token != os.getenv('REBUILD_TOKEN'):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        print("Running get_knowledge_base.py script...")
        result = subprocess.run(["python3", "get_knowledge_base.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running get_knowledge_base.py script: {result.stderr}")
            return jsonify({"error": "Error running get_knowledge_base.py script", "details": result.stderr}), 500

        print("Finished running get_knowledge_base.py script.")

        # get Dockerfiles and compose files from samples repo
        print("Running get_samples_examples.py script...")
        result = subprocess.run(["python3", "get_samples_examples.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running get_samples_examples.py script: {result.stderr}")
            return jsonify({"error": "Error running get_samples_examples.py script", "details": result.stderr}), 500

        print("Finished running get_samples_examples.py script.")

        print("Rebuilding embeddings...")
        try:
            rag_system.rebuild_embeddings()
        except Exception as e:
            print(f"Error rebuilding embeddings: {str(e)}")
            return jsonify({"error": "Error rebuilding embeddings", "details": str(e)}), 500

        print("Finished rebuilding embeddings.")
        return jsonify({"status": "Rebuild triggered successfully"}), 200

    except Exception as e:
        print(f"Error in /trigger-rebuild endpoint: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/data/<path:name>")
@csrf.exempt
def download_file(name):
    return send_from_directory(
        "data", name, as_attachment=True
    )

if os.getenv('DEBUG') == '1':
    @app.route('/ask/debug', methods=['POST'])
    def debug_context():
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "Query is required"}), 400
        context = rag_system.get_context(query)
        return jsonify({"context": context})


# Retrieve a conversation from Intercom API by its ID
def fetch_intercom_conversation(conversation_id):
    id = conversation_id
    url = "https://api.intercom.io/conversations/" + id
    token = os.getenv('INTERCOM_TOKEN')
    if not token:
        return jsonify({"error": "Intercom token not set"}), 500

    headers = {
        "Content-Type": "application/json",
        "Intercom-Version": "2.13",
        "Authorization": "Bearer " + token
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to fetch conversation {id} from Intercom; status code: {response.status_code}, response: {response.text}")
        return jsonify({"error": "Failed to fetch conversation from Intercom"}), response.status_code
    
    return response

# Determines the user query from the Intercom conversation response
def get_user_query(response, conversation_id):
    # Extract conversation parts from an Intercom request response
    result = extract_conversation_parts(response)
    logger.info(f"Extracted {len(result)} parts from conversation {conversation_id}")

    # Get and join the latest user messages from the conversation parts
    joined_text = extract_latest_user_messages(result)
    if not joined_text:
        return jsonify({"info": "No entries made by user found."}), 204
    return joined_text

# Extract conversation parts into a simplified JSON format
def extract_conversation_parts(response):
    data = response.json()
    parts = data.get('conversation_parts', {}).get('conversation_parts', [])
    extracted_parts = []
    for part in parts:
        body = part.get('body', '')
        if not body:
            continue
        author = part.get('author', {})
        created_at = part.get('created_at')
        extracted_parts.append({'body': body, 'author': author, 'created_at': created_at})
    return extracted_parts

# Joins the latest user entries in the conversation starting from the last non-user (i.e. admin) entry
def extract_latest_user_messages(conversation_parts):
    # Find the index of the last non-user entry
    last_non_user_idx = None
    for idx in range(len(conversation_parts) - 1, -1, -1):
        if conversation_parts[idx].get('author', {}).get('type') != 'user':
            last_non_user_idx = idx
            break

    # Collect user entries after the last non-user entry
    if last_non_user_idx is not None:
        last_user_entries = [
            part for part in conversation_parts[last_non_user_idx + 1 :]
            if part.get('author', {}).get('type') == 'user'
        ]
    else:
        # If there is no non-user entry, include all user entries
        last_user_entries = [
            part for part in conversation_parts if part.get('author', {}).get('type') == 'user'
        ]

    # If no user entries found, return None
    if not last_user_entries:
        return None

    # Only keep the 'body' field from each user entry
    bodies = [part['body'] for part in last_user_entries if 'body' in part]

    # Parse and concatenate all user message bodies as plain text
    parsed_bodies = []
    for html_body in bodies:
        parsed_bodies.append(parse_html_to_text(html_body))

    # Join all parsed user messages into a single string
    joined_text = " ".join(parsed_bodies)
    return joined_text

# Helper function to parse HTML into plain text
def parse_html_to_text(html_content):
    parser = BodyHTMLParser()
    parser.feed(html_content)
    return parser.get_text()

# Store conversation ID in persistent storage
def set_conversation_human_replied(conversation_id):
    try:
        # Use a Redis set to avoid duplicates
        r.set(conversation_id, '1', ex=60*60*24) # Set TTL expiration to 1 day
        logger.info(f"Added conversation_id {conversation_id} to Redis set admin_replied_conversations")
    except Exception as e:
        logger.error(f"Error adding conversation_id to Redis: {e}")
    
# Check if a conversation is already marked as replied by a human admin
def is_conversation_human_replied(conversation_id):
    try:
        return r.exists(conversation_id)
    except Exception as e:
        logger.error(f"Error checking conversation_id in Redis: {e}")
        return False


# Post a reply to a conversation through Intercom API
def post_intercom_reply(conversation_id, response_text):
    url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    token = os.getenv('INTERCOM_TOKEN')
    if not token:
        return jsonify({"error": "Intercom token not set"}), 500

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
    }

    payload = {
        "message_type": "comment",
        "type": "admin",
        "admin_id": int(os.getenv('INTERCOM_ADMIN_ID')),
        "body": response_text
    }

    response = requests.post(url, json=payload, headers=headers)
    logger.info(f"Posted reply to Intercom; response status code: {response.status_code}")

    return response.json(), response.status_code


# Returns a generated LLM answer to the Intercom conversation based on previous user message history
def answer_intercom_conversation(conversation_id):
    logger.info(f"Received request to get conversation {conversation_id}")

    # Retrieves the history of the conversation thread in Intercom
    conversation= fetch_intercom_conversation(conversation_id)

    # Extracts the user query (which are latest user messages joined into a single string) from conversation history
    user_query = get_user_query(conversation, conversation_id)
    logger.info(f"Joined user messages: {user_query}")

    # Use a deterministic, non-reversible hash for anonymous_id for Intercom conversations
    anon_hash = hashlib.sha256(f"intercom-{conversation_id}".encode()).hexdigest()
    
    # Generate the exact response using the RAG system
    llm_response = "".join(generate(user_query, 'Intercom Conversation', anon_hash))
    llm_response = llm_response + " 🤖" # Add a marker to indicate the end of the response

    logger.info(f"LLM response: {llm_response}")

    return post_intercom_reply(conversation_id, llm_response)

# Endpoint to handle incoming webhooks from Intercom
@app.route('/intercom-webhook', methods=['POST'])
@csrf.exempt
def handle_webhook():
    data = request.json

    logger.info(f"Received Intercom webhook: {data}")
    conversation_id = data.get('data', {}).get('item', {}).get('id')

    # Check for the type of the webhook event
    topic = data.get('topic')
    logger.info(f"Webhook topic: {topic}")
    if topic == 'conversation.admin.replied':

        # Check if the admin is a bot or human based on presence of a message marker (e.g., "🤖") in the last message
        last_message = data.get('data', {}).get('item', {}).get('conversation_parts', {}).get('conversation_parts', [])[-1].get('body', '')
        last_message_text = parse_html_to_text(last_message)

        logger.info(f"Parsed last message text: {last_message_text}")
        if last_message_text and last_message_text.endswith("🤖"):
            # If the last message ends with the marker, it indicates a bot reply
            logger.info(f"Last message in conversation {conversation_id} ends with the marker 🤖")
            logger.info(f"Detected bot admin reply in conversation {conversation_id}; no action taken.")
        else:
            # If the last message does not end with the marker, it indicates a human reply
            logger.info(f"Detected human admin reply in conversation {conversation_id}; marking as human admin-replied...")
            # Mark the conversation as replied by a human admin to skip LLM responses in the future
            set_conversation_human_replied(conversation_id)
            logger.info(f"Successfully marked conversation {conversation_id} as human admin-replied.")
        return 'OK'
    elif topic == 'conversation.user.replied':
        # In this case, the webhook event is a user reply, not an admin reply
        # Check if the conversation was replied previously by a human admin
        if is_conversation_human_replied(conversation_id):
            logger.info(f"Conversation {conversation_id} already marked as human admin-replied; no action taken.")
            return 'OK'
        # Fetch the conversation and generate an LLM answer for the user
        logger.info(f"Detected a user reply in conversation {conversation_id}; fetching an answer from LLM...")
        answer_intercom_conversation(conversation_id)
    else:
        logger.info(f"Received webhook for unsupported topic: {topic}; no action taken.")
    return 'OK'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
