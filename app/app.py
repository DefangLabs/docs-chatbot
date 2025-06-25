from flask import Flask, request, jsonify, render_template, Response, stream_with_context, session, send_from_directory
from flask_wtf.csrf import CSRFProtect
from rag_system import rag_system
import hashlib
import subprocess
import os
import segment.analytics as analytics
import uuid

from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

import logging
import redis
from intercom import parse_html_to_text, set_conversation_human_replied, is_conversation_human_replied, answer_intercom_conversation, check_intercom_ip
from utils import generate

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


# Handle incoming webhooks from Intercom
@app.route('/intercom-webhook', methods=['POST'])
@csrf.exempt
def handle_webhook():
    if not check_intercom_ip(request):
        return jsonify({"error": "Unauthorized IP"}), 403

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
            set_conversation_human_replied(conversation_id, r)
            logger.info(f"Successfully marked conversation {conversation_id} as human admin-replied.")
    elif topic == 'conversation.user.replied':
        # In this case, the webhook event is a user reply, not an admin reply
        # Check if the conversation was replied previously by a human admin
        if is_conversation_human_replied(conversation_id, r):
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
