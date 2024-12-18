from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_wtf.csrf import CSRFProtect
from rag_system import rag_system
import subprocess
app = Flask(__name__, static_folder='templates/images')

import os

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = bool(os.getenv('SESSION_COOKIE_SECURE'))

csrf = CSRFProtect(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    def generate():
        try:
            for token in rag_system.answer_query_stream(query):
                yield token
        except Exception as e:
            print(f"Error in /ask endpoint: {e}")
            yield "Internal Server Error"

    return Response(stream_with_context(generate()), content_type='text/markdown')

@app.route('/trigger-rebuild', methods=['POST'])
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)