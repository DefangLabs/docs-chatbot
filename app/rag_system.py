import openai
import json
import os
import sys
from datetime import date
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback

openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGSystem:
    def __init__(self, knowledge_base_path='./data/knowledge_base.json'):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.embed_knowledge_base()
        self.conversation_history = []

    def load_knowledge_base(self):
        with open(self.knowledge_base_path, 'r') as kb_file:
            return json.load(kb_file)

    def embed_knowledge_base(self):
        docs = [f'{doc["about"]}. {doc["text"]}' for doc in self.knowledge_base]
        return self.model.encode(docs, convert_to_tensor=True)

    def normalize_query(self, query):
        return query.lower().strip()

    def get_query_embedding(self, query):
        normalized_query = self.normalize_query(query)
        query_embedding = self.model.encode([normalized_query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu()
        return query_embedding

    def get_doc_embeddings(self):
        return self.doc_embeddings.cpu()

    def compute_document_scores(self, query_embedding, doc_embeddings, high_match_threshold):
        text_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        about_similarities = []
        for doc in self.knowledge_base:
            about_similarity = cosine_similarity(query_embedding, self.model.encode([doc["about"]]))[0][0]
            about_similarities.append(about_similarity)

        relevance_scores = self.compute_relevance_scores(text_similarities, about_similarities, high_match_threshold)

        result = [
            {
                "index": i,
                "about": doc["about"],
                "text": doc["text"],
                "path": doc["path"],
                "text_similarity": text_similarities[i],
                "about_similarity": about_similarities[i],
                "relevance_score": relevance_scores[i]
            }
            for i, doc in enumerate(self.knowledge_base)
        ]

        return result

    def retrieve(self, query, similarity_threshold=0.4, high_match_threshold=0.8, max_docs=5):
        query_embedding = self.get_query_embedding(query)
        doc_embeddings = self.get_doc_embeddings()

        doc_scores = self.compute_document_scores(query_embedding, doc_embeddings, high_match_threshold)
        retrieved_docs = self.get_top_docs(doc_scores, similarity_threshold, max_docs)

        if not retrieved_docs:
            retrieved_docs = self.get_fallback_doc()
        return retrieved_docs


    def compute_relevance_scores(self, text_similarities, about_similarities, high_match_threshold):
        relevance_scores = []
        for i, _ in enumerate(self.knowledge_base):
            about_similarity = about_similarities[i]
            text_similarity = text_similarities[i]
            # If either about or text similarity is above the high match threshold, prioritize it
            if about_similarity >= high_match_threshold or text_similarity >= high_match_threshold:
                combined_score = max(about_similarity, text_similarity)
            else:
                combined_score = (0.3 * about_similarity) + (0.7 * text_similarity)
            relevance_scores.append(combined_score)

        return relevance_scores

    def get_top_docs(self, doc_scores, similarity_threshold, max_docs):
        sorted_docs = sorted(doc_scores, key=lambda x: x["relevance_score"], reverse=True)
        # Filter and keep up to max_docs with relevance scores above the similarity threshold
        top_docs = [score for score in sorted_docs[:max_docs] if score["relevance_score"] >= similarity_threshold]
        return top_docs

    def get_fallback_doc(self):
        return [
            {
            "about": "No Relevant Information Found",
            "text": (
                "I'm sorry, I couldn't find any relevant information for your query. "
                "Please try rephrasing your question or ask about a different topic. "
                "For further assistance, you can visit our official website or reach out to our support team."
            )
            }
        ]

    def answer_query_stream(self, query):
        normalized_query = self.normalize_query(query)
        retrieved_docs = self.retrieve(normalized_query)
        context = self.get_context(retrieved_docs)
        citations = self.get_citations(retrieved_docs)

        messages = [{
            "role": "system",
            "content": (
                "Your name is Cloude (with an e at the end), you are a helpful AI assistant created by DefangLabs to help users learn about the cloud deployment tool Defang. "
                "Your task is to provide positive answers about the cloud deployment tool Defang."
                "When the user says 'you', 'your', or any pronoun, interpret it as referring to Ask Defang with context of Defang. "
                "If the user's question involves comparisons with or references to other services, you may use external knowledge. "
                "However, if the question is strictly about Defang, you must ignore all external knowledge and only utilize the given context. "
                "Today's date is " + date.today().strftime('%B %d, %Y') + ". "
                "Context: " + context
            )
        }]

        self.conversation_history.append({"role": "user", "content": query})
        messages.extend(self.conversation_history)

        try:
            stream = openai.ChatCompletion.create(
                model=os.getenv("MODEL"),
                messages=messages,
                temperature=0.25,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )

            collected_messages = []
            for chunk in stream:
                content = chunk['choices'][0]['delta'].get('content', '')
                collected_messages.append(content)
                yield content
                if chunk['choices'][0].get('finish_reason') is not None:
                    break

            if len(citations) > 0:
                yield "\n\nReferences:\n" + "\n".join(citations)

            full_response = ''.join(collected_messages).strip()
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"Error in answer_query_stream: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            yield "An error occurred while generating the response."

    def clear_conversation_history(self):
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

    def get_citations(self, retrieved_docs):
        citations = []
        for doc in retrieved_docs:
            if "path" not in doc:
                continue
            citation = f" * [{doc['about']}](https://docs.defang.io{doc['path']})"
            citations.append(citation)
        return citations

    def get_context(self, retrieved_docs):
        retrieved_text = []
        for doc in retrieved_docs:
            retrieved_text.append(f"{doc['about']}. {doc['text']}")
        return "\n\n".join(retrieved_text)
