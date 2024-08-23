from flask import Flask, request, jsonify, render_template
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
    

# New endpoint for triggering the rebuild
@app.route('/trigger-rebuild', methods=['POST'])
def trigger_rebuild():
    token = request.args.get('token')
    if token != 'b75c82e0-2f47-4fcb-8b02-d66932803885 ':  # Replace with your actual token
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # Call the function to rebuild embeddings or whatever process is needed
        rag_system.rebuild_embeddings()
        return jsonify({"status": "Rebuild triggered successfully"}), 200
    except Exception as e:
        print(f"Error in /trigger-rebuild endpoint: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
