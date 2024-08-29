from flask import Flask, request, jsonify, render_template
import subprocess
import os
from rag_system import rag_system

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return render_template('index.html', query=None, response="No query provided")
        
        try:
            response = rag_system.answer_query(query)
            return render_template('index.html', query=query, response=response)
        except Exception as e:
            print(f"Error in /ask endpoint: {e}")
            return render_template('index.html', query=query, response="Internal Server Error")
    return render_template('index.html', query=None, response=None)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        response = rag_system.answer_query(query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    

# # New endpoint for triggering the rebuild
# def run_get_knowledge_base_script():
#     """ Function to run the get_knowledge_base.py script from the parent directory """
#     try:
#         subprocess.run(['python', 'get_knowledge_base.py'], check=True)

#     except subprocess.CalledProcessError as e:
#         print(f"Error running get_knowledge_base.py: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

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
