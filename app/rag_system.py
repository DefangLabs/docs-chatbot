import openai
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import redis

# Ensure you have set the OPENAI_API_KEY in your environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGSystem:
    def __init__(self, redis_host=None, redis_port=None):
        # Use environment variables to determine Redis host and port
        if redis_host is None:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
        if redis_port is None:
            redis_port = int(os.getenv('REDIS_PORT', 6379))
        
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load knowledge base and embeddings
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """
        Load the knowledge base from a JSON file and store each document with its embedding in Redis under a single key.
        """
        knowledge_base_keys = self.redis_client.keys('doc:*')

        if not knowledge_base_keys:
            print("No knowledge base found in Redis. Loading from JSON and storing in Redis...")

            with open('knowledge_base.json', 'r') as kb_file:
                knowledge_base = json.load(kb_file)

                # Store each document with its embedding in Redis under a single key
                for doc in knowledge_base:
                    doc_key = f'doc:{doc["id"]}'
                    doc_embedding = self.model.encode(f'{doc["about"]}. {doc["text"]}').tolist()

                    # Store everything as a single JSON object
                    self.redis_client.set(doc_key, json.dumps({
                        "about": doc["about"],
                        "text": doc["text"],
                        "embedding": doc_embedding
                    }))

        else:
            print("Knowledge base already loaded in Redis.")

    def get_doc_by_id(self, doc_id):
        """
        Retrieve a document and its embedding by its ID from Redis.
        """
        doc_data = self.redis_client.get(f'doc:{doc_id}')
        if doc_data:
            return json.loads(doc_data)
        else:
            return None

    def normalize_query(self, query):
        """
        Normalize the query by converting it to lowercase and stripping whitespace.
        """
        return query.lower().strip()

    def retrieve(self, query, similarity_threshold=0.7, high_match_threshold=0.8, max_docs=5):
        # Normalize query
        normalized_query = self.normalize_query(query)
        print(f"Retrieving context for query: '{normalized_query}'")

        # Query embedding
        query_embedding = self.model.encode([normalized_query])[0]

        # Initialize relevance scores
        relevance_scores = []

        for doc_key in self.redis_client.keys('doc:*'):
            doc_id = doc_key.split(":")[1]
            doc_data = self.get_doc_by_id(doc_id)

            if not doc_data:
                continue

            doc_embedding = np.array(doc_data["embedding"])

            # Calculate similarities
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]

            # Calculate about similarity separately
            about_similarity = cosine_similarity([query_embedding], self.model.encode([doc_data["about"]]))[0][0]

            # Give more weight to text similarity
            combined_score = (0.3 * about_similarity) + (0.7 * similarity)

            # If either about or text similarity is above the high match threshold, prioritize it
            if about_similarity >= high_match_threshold or similarity >= high_match_threshold:
                combined_score = max(about_similarity, similarity)

            relevance_scores.append((doc_id, combined_score))

        # Sort by combined score in descending order
        sorted_indices = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
        top_indices = [doc_id for doc_id, score in sorted_indices[:max_docs] if score >= similarity_threshold]

        # Retrieve the most relevant documents, including both 'about' and 'text' fields
        retrieved_docs = [f'{self.get_doc_by_id(doc_id)["about"]}. {self.get_doc_by_id(doc_id)["text"]}' for doc_id in top_indices]

        if not retrieved_docs:
            max_index = np.argmax([score for _, score in relevance_scores])
            max_doc_id = relevance_scores[max_index][0]
            retrieved_docs.append(f'{self.get_doc_by_id(max_doc_id)["about"]}. {self.get_doc_by_id(max_doc_id)["text"]}')

        context = "\n\n".join(retrieved_docs)
        print("Retrieved Context:\n", context)

        return context

    def generate_response(self, query, context):
        # Normalize query
        normalized_query = self.normalize_query(query)
        print(f"Generating response for query: '{normalized_query}'")

        try:
            prompt = (
                "You are a dedicated assistant designed to provide positive answers about Defang. "
                "When the user says 'you', 'your', or any pronoun, interpret it as referring to Defang with context of Defang also. "
                "If the user's question involves comparisons with or references to other services, you may use external knowledge. "
                "However, if the question is strictly about Defang, you must ignore all external knowledge and only utilize the given context. "
                "When generating the answer, please put the answer first and the justification later. "
                "Any mentions of BYOD means BRING YOUR OWN DOMAIN and NOT BRING YOUR OWN DEVICE."
                "Your objective is to remain strictly within the confines of the given context unless comparisons to other services are explicitly mentioned. "
                "Although this rarely happens, if the prompt is not related to defang reply with prompt out of scope. If the prompt contains the word `defang` proceed with answering"
                "\n\nContext:\n" + context + "\n\n"
                "User Question: " + query + "\n\n"
                "Answer:"
            )

            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": normalized_query}
                ],
                temperature=0.05,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Print the response generated by the model
            generated_response = response['choices'][0]['message']['content'].strip()

            print("Generated Response:\n", generated_response)

            return generated_response

        except openai.error.OpenAIError as e:
            print(f"Error generating response from OpenAI: {e}")
            return "An error occurred while generating the response."

    def answer_query(self, query):
        try:
            # Normalize query before use
            normalized_query = self.normalize_query(query)
            context = self.retrieve(normalized_query)
            response = self.generate_response(normalized_query, context)
            return response
        except Exception as e:
            print(f"Error in answer_query: {e}")
            return "An error occurred while generating the response."

    def rebuild_embeddings(self):
        """
        Rebuild the embeddings for the knowledge base. This should be called whenever the knowledge base is updated.
        """
        print("Rebuilding embeddings for the knowledge base...")
        self.load_knowledge_base()  # Reload the knowledge base and rebuild the embeddings
        print("Embeddings have been rebuilt.")

# Instantiate the RAGSystem
rag_system = RAGSystem()
