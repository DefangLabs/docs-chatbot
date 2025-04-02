from flask import Flask, request, jsonify, render_template, Response, stream_with_context, session
from flask_wtf.csrf import CSRFProtect
from rag_system import rag_system
import hashlib
import subprocess
import os
import segment.analytics as analytics
import uuid
import sys
import traceback

analytics.write_key = os.getenv('SEGMENT_WRITE_KEY')

app = Flask(__name__, static_folder='templates/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = bool(os.getenv('SESSION_COOKIE_SECURE'))

csrf = CSRFProtect(app)


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

    def generate():
        full_response = ""
        try:
            for token in rag_system.answer_query_stream(query):
                yield token
                full_response += token
        except Exception as e:
            print(f"Error in /ask endpoint: {e}", file=sys.stderr)
            traceback.print_exc()
            yield "Internal Server Error"

        if not full_response:
            full_response = "No response generated"

        if analytics.write_key:
            # Track the query and response
            analytics.track(
                anonymous_id=anonymous_id,
                event='Chatbot Question submitted',
                properties={'query': query, 'response': full_response, 'source': 'Ask Defang'}
            )

    return Response(stream_with_context(generate()), content_type='text/markdown')

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
    if token != 'b75c82e0-2f47-4fcb-8b02-d66932803885':
        return jsonify({"error": "Unauthorized"}), 401
    try:
        print("Running get_knowledge_base.py script...")
        result = subprocess.run(["python3", "get_knowledge_base.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running get_knowledge_base.py script: {result.stderr}")
            return jsonify({"error": "Error running get_knowledge_base.py script", "details": result.stderr}), 500

        print("Finished running get_knowledge_base.py script.")

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

if os.getenv('DEBUG') == '1':
    @app.route('/ask/debug', methods=['POST'])
    def debug_context():
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "Query is required"}), 400
        context = rag_system.get_context(query)
        return jsonify({"context": context})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
