import openai
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure you have set the OPENAI_API_KEY in your environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGSystem:
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.embed_knowledge_base()
        self.conversation_history = []  # To store the conversation history

    def load_knowledge_base(self):
        """
        Load the knowledge base from a JSON file.
        """
        with open(self.knowledge_base_path, 'r') as kb_file:
            return json.load(kb_file)

    def embed_knowledge_base(self):
        """
        Embed the knowledge base using the SentenceTransformer model.
        Combines 'about' and 'text' fields for each document for embedding.
        """
        docs = [f'{doc["about"]}. {doc["text"]}' for doc in self.knowledge_base]
        return self.model.encode(docs, convert_to_tensor=True)

    def normalize_query(self, query):
        """
        Normalize the query by converting it to lowercase and stripping whitespace.
        """
        return query.lower().strip()

    def retrieve(self, query, similarity_threshold=0.7, high_match_threshold=0.8, max_docs=5):
        """
        Retrieve relevant documents from the knowledge base using cosine similarity.
        """
        normalized_query = self.normalize_query(query)
        print(f"Retrieving context for query: '{normalized_query}'")

        # Query embedding
        query_embedding = self.model.encode([normalized_query], convert_to_tensor=True)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Initialize relevance scores
        relevance_scores = []

        for i, doc in enumerate(self.knowledge_base):
            about_similarity = cosine_similarity(query_embedding, self.model.encode([doc["about"]]))[0][0]
            text_similarity = similarities[i]
            
            combined_score = (0.3 * about_similarity) + (0.7 * text_similarity)
            if about_similarity >= high_match_threshold or text_similarity >= high_match_threshold:
                combined_score = max(about_similarity, text_similarity)
                
            relevance_scores.append((i, combined_score))

        sorted_indices = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, score in sorted_indices[:max_docs] if score >= similarity_threshold]

        retrieved_docs = [f'{self.knowledge_base[i]["about"]}. {self.knowledge_base[i]["text"]}' for i in top_indices]

        if not retrieved_docs:
            max_index = np.argmax(similarities)
            retrieved_docs.append(f'{self.knowledge_base[max_index]["about"]}. {self.knowledge_base[max_index]["text"]}')

        context = "\n\n".join(retrieved_docs)
        print("Retrieved Context:\n", context)

        return context

    def generate_response(self, query, context):
        """
        Generate a response from OpenAI using conversation history and knowledge base context.
        """
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Include conversation history in the prompt
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.extend(self.conversation_history)  # Include past conversation history

        # Prepare the context from the knowledge base
        system_message = {
            "role": "system",
            "content": (
                "You are a dedicated assistant designed to provide positive answers about Defang. "
                "When the user says 'you', 'your', or any pronoun, interpret it as referring to Defang with context of Defang. "
                "If the user's question involves comparisons with or references to other services, you may use external knowledge. "
                "However, if the question is strictly about Defang, you must ignore all external knowledge and only utilize the given context. "
                "Context: " + context
            )
        }

        messages.append(system_message)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            generated_response = response['choices'][0]['message']['content'].strip()

            # Add the bot's response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": generated_response})

            print("Generated Response:\n", generated_response)
            return generated_response

        except openai.error.OpenAIError as e:
            print(f"Error generating response from OpenAI: {e}")
            return "An error occurred while generating the response."

    def answer_query(self, query):
        """
        Answer the user query, leveraging knowledge base context and conversation history.
        """
        try:
            normalized_query = self.normalize_query(query)
            context = self.retrieve(normalized_query)
            response = self.generate_response(normalized_query, context)
            return response
        except Exception as e:
            print(f"Error in answer_query: {e}")
            return "An error occurred while generating the response."

    def clear_conversation_history(self):
        """
        Clear the stored conversation history.
        This can be called to reset the conversation for a new session.
        """
        self.conversation_history = []
        print("Conversation history cleared.")

    def rebuild_embeddings(self):
        """
        Rebuild the embeddings for the knowledge base. This should be called whenever the knowledge base is updated.
        """
        print("Rebuilding embeddings for the knowledge base...")
        self.knowledge_base = self.load_knowledge_base()  # Reload the knowledge base
        self.doc_embeddings = self.embed_knowledge_base()  # Rebuild the embeddings
        print("Embeddings have been rebuilt.")

# Instantiate the RAGSystem
rag_system = RAGSystem()
