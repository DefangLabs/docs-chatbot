# Intercom API helper functions for handling conversations and replies
import os
import requests
import hashlib
from flask import jsonify
from html.parser import HTMLParser

import logging
logger = logging.getLogger(__name__)

from utils import generate

class BodyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return ''.join(self.text)

# Retrieve a conversation from Intercom API by its ID
def fetch_intercom_conversation(conversation_id):
    url = "https://api.intercom.io/conversations/" + conversation_id
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
        logger.error(f"Failed to fetch conversation {conversation_id} from Intercom; status code: {response.status_code}, response: {response.text}")
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
        return "No entries made by user found.", 204
    return joined_text, 200

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
def set_conversation_human_replied(conversation_id, redis_client):
    try:
        # Use a Redis set to avoid duplicates
        redis_client.set(conversation_id, '1', ex=60*60*24) # Set TTL expiration to 1 day
        logger.info(f"Added conversation_id {conversation_id} to Redis set admin_replied_conversations")
    except Exception as e:
        logger.error(f"Error adding conversation_id to Redis: {e}")
    
# Check if a conversation is already marked as replied by a human admin
def is_conversation_human_replied(conversation_id, redis_client):
    try:
        return redis_client.exists(conversation_id)
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
    conversation = fetch_intercom_conversation(conversation_id)
    # If a tuple is returned, it is an error response
    if isinstance(conversation, tuple):
        return conversation

    # Extracts the user query (which are latest user messages joined into a single string) from conversation history
    user_query, status_code = get_user_query(conversation, conversation_id)
    # If a tuple is returned, it is an error response
    if status_code != 200:
        return jsonify(user_query), status_code

    logger.info(f"Joined user messages: {user_query}")

    # Use a deterministic, non-reversible hash for anonymous_id for Intercom conversations
    anon_hash = hashlib.sha256(f"intercom-{conversation_id}".encode()).hexdigest()
    
    # Generate the exact response using the RAG system
    llm_response = "".join(generate(user_query, 'Intercom Conversation', anon_hash))
    llm_response = llm_response + " 🤖" # Add a marker to indicate the end of the response

    logger.info(f"LLM response: {llm_response}")

    return post_intercom_reply(conversation_id, llm_response)

def check_intercom_ip(request):
        # Restrict webhook access to a list of allowed IP addresses
    INTERCOM_ALLOWED_IPS = [
        "34.231.68.152",
        "34.197.76.213",
        "35.171.78.91",
        "35.169.138.21",
        "52.70.27.159",
        "52.44.63.161"
    ]
    remote_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    # X-Forwarded-For may contain a comma-separated list; take the first IP
    remote_ip = remote_ip.split(',')[0].strip() if remote_ip else None

    if remote_ip not in INTERCOM_ALLOWED_IPS:
        # logger.info(f"Rejected webhook from unauthorized IP: {remote_ip}")
        return False
    
    return True
