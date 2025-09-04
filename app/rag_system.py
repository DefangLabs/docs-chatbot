import openai
import json
import os
import sys
import logging
import threading
from datetime import date
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from atomicwrites import atomic_write


openai.api_base = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")


class RAGSystem:
    # Cache file paths
    DOC_EMBEDDINGS_PATH = "./data/doc_embeddings.npy"
    DOC_ABOUT_EMBEDDINGS_PATH = "./data/doc_about_embeddings.npy"

    def __init__(self, knowledge_base_path="./data/knowledge_base.json"):
        self._update_lock = threading.Lock()
        self.knowledge_base_path = knowledge_base_path

        knowledge_base = self.load_knowledge_base()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # load existing embeddings if available
        logging.info("Embedding knowledge base...")

        if os.path.exists(self.DOC_ABOUT_EMBEDDINGS_PATH) and os.path.exists(
            self.DOC_EMBEDDINGS_PATH
        ):
            with self._update_lock:
                self.doc_about_embeddings = np.load(self.DOC_ABOUT_EMBEDDINGS_PATH)
                logging.info(
                    "Loaded existing about document about embeddings from disk."
                )
                self.doc_embeddings = np.load(self.DOC_EMBEDDINGS_PATH)
                logging.info("Loaded existing document embeddings from disk.")
                self.knowledge_base = knowledge_base

                # Save file timestamps when loading cache
                self.doc_embeddings_timestamp = os.path.getmtime(
                    self.DOC_EMBEDDINGS_PATH
                )
                self.doc_about_embeddings_timestamp = os.path.getmtime(
                    self.DOC_ABOUT_EMBEDDINGS_PATH
                )
                logging.info(
                    f"Cache loaded - doc_embeddings timestamp: {self.doc_embeddings_timestamp}, doc_about_embeddings timestamp: {self.doc_about_embeddings_timestamp}"
                )
        else:
            self.rebuild_embeddings(knowledge_base)

        logging.info("Knowledge base embeddings created")
        self.conversation_history = []

    def _atomic_save_numpy(self, file_path, data):
        with atomic_write(file_path, mode="wb", overwrite=True) as f:
            np.save(f, data)

    def rebuild_embeddings(self, knowledge_base):
        logging.info("Rebuilding document embeddings...")

        new_doc_embeddings = self.embed_knowledge_base(knowledge_base)
        new_about_embeddings = self.embed_knowledge_base_about(knowledge_base)

        # Defensive check for size mismatches
        sizes = [
            len(new_about_embeddings),
            len(new_doc_embeddings),
            len(knowledge_base),
        ]
        if len(set(sizes)) > 1:  # Not all sizes are equal
            logging.error(
                f"rebuild embeddings Array size mismatch detected: text_similarities={sizes[0]}, about_similarities={sizes[1]}, knowledge_base={sizes[2]}"
            )
            return  # Abandon update

        # Atomically update files, in-memory cache, and timestamps
        with self._update_lock:
            self._atomic_save_numpy(
                self.DOC_EMBEDDINGS_PATH, new_doc_embeddings.cpu().numpy()
            )
            self._atomic_save_numpy(
                self.DOC_ABOUT_EMBEDDINGS_PATH, new_about_embeddings.cpu().numpy()
            )
            self.knowledge_base = knowledge_base
            self.doc_embeddings = new_doc_embeddings
            self.doc_about_embeddings = new_about_embeddings
            self.doc_embeddings_timestamp = os.path.getmtime(self.DOC_EMBEDDINGS_PATH)
            self.doc_about_embeddings_timestamp = os.path.getmtime(
                self.DOC_ABOUT_EMBEDDINGS_PATH
            )

        logging.info("Embeddings rebuilt successfully.")

    def load_knowledge_base(self):
        with open(self.knowledge_base_path, "r") as kb_file:
            return json.load(kb_file)

    def embed_knowledge_base(self, knowledge_base):
        docs = [f"{doc['about']}. {doc['text']}" for doc in knowledge_base]
        return self.model.encode(docs, convert_to_tensor=True)

    def embed_knowledge_base_about(self, knowledge_base):
        return self.model.encode(
            [doc["about"] for doc in knowledge_base], convert_to_tensor=True
        )

    def normalize_query(self, query):
        return query.lower().strip()

    def get_query_embedding(self, query):
        normalized_query = self.normalize_query(query)
        query_embedding = self.model.encode([normalized_query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu()
        return query_embedding

    def get_doc_embeddings(self):
        return self.doc_embeddings

    def get_doc_about_embeddings(self):
        return self.doc_about_embeddings

    def compute_document_scores(
        self,
        query_embedding,
        doc_embeddings,
        doc_about_embeddings,
        high_match_threshold,
    ):
        text_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        about_similarities = cosine_similarity(query_embedding, doc_about_embeddings)[0]
        relevance_scores = self.compute_relevance_scores(
            text_similarities, about_similarities, high_match_threshold
        )

        result = [
            {
                "index": i,
                "about": doc["about"],
                "text": doc["text"],
                "path": doc["path"],
                "text_similarity": text_similarities[i],
                "about_similarity": about_similarities[i],
                "relevance_score": relevance_scores[i],
            }
            for i, doc in enumerate(self.knowledge_base)
        ]

        return result

    def cache_check(func):
        """Decorator to automatically check cache consistency"""

        def wrapper(self, *args, **kwargs):
            try:
                current_times = [
                    os.path.getmtime(self.DOC_EMBEDDINGS_PATH),
                    os.path.getmtime(self.DOC_ABOUT_EMBEDDINGS_PATH),
                ]
                stored_times = [
                    self.doc_embeddings_timestamp,
                    self.doc_about_embeddings_timestamp,
                ]

                # update cache if timestamps are different from out last load
                if current_times != stored_times:
                    self._reload_cache()

            except (OSError, FileNotFoundError, PermissionError):
                logging.warning("Cache files inaccessible, rebuilding...")
                self.rebuild_embeddings()

            return func(self, *args, **kwargs)

        return wrapper

    def _reload_cache(self):
        self.doc_embeddings = np.load(self.DOC_EMBEDDINGS_PATH)
        self.doc_about_embeddings = np.load(self.DOC_ABOUT_EMBEDDINGS_PATH)

        # update our timestamps of the cached files
        self.doc_embeddings_timestamp = os.path.getmtime(self.DOC_EMBEDDINGS_PATH)
        self.doc_about_embeddings_timestamp = os.path.getmtime(
            self.DOC_ABOUT_EMBEDDINGS_PATH
        )

    @cache_check
    def retrieve(
        self, query, similarity_threshold=0.4, high_match_threshold=0.8, max_docs=5
    ):
        query_embedding = self.get_query_embedding(query)
        doc_embeddings = self.get_doc_embeddings()
        doc_about_embeddings = self.get_doc_about_embeddings()

        doc_scores = self.compute_document_scores(
            query_embedding, doc_embeddings, doc_about_embeddings, high_match_threshold
        )
        retrieved_docs = self.get_top_docs(doc_scores, similarity_threshold, max_docs)

        if not retrieved_docs:
            retrieved_docs = self.get_fallback_doc()
        return retrieved_docs

    def compute_relevance_scores(
        self, text_similarities, about_similarities, high_match_threshold
    ):
        relevance_scores = []

        for i, _ in enumerate(self.knowledge_base):
            about_similarity = about_similarities[i]
            text_similarity = text_similarities[i]
            # If either about or text similarity is above the high match threshold, prioritize it
            if (
                about_similarity >= high_match_threshold
                or text_similarity >= high_match_threshold
            ):
                combined_score = max(about_similarity, text_similarity)
            else:
                combined_score = (0.3 * about_similarity) + (0.7 * text_similarity)
            relevance_scores.append(combined_score)

        return relevance_scores

    def get_top_docs(self, doc_scores, similarity_threshold, max_docs):
        sorted_docs = sorted(
            doc_scores, key=lambda x: x["relevance_score"], reverse=True
        )
        # Filter and keep up to max_docs with relevance scores above the similarity threshold
        top_docs = [
            score
            for score in sorted_docs[:max_docs]
            if score["relevance_score"] >= similarity_threshold
        ]
        return top_docs

    def get_fallback_doc(self):
        return [
            {
                "about": "No Relevant Information Found",
                "text": (
                    "I'm sorry, I couldn't find any relevant information for your query. "
                    "Please try rephrasing your question or ask about a different topic. "
                    "For further assistance, you can visit our official website or reach out to our support team."
                ),
            }
        ]

    def answer_query_stream(self, query):
        normalized_query = self.normalize_query(query)
        retrieved_docs = self.retrieve(normalized_query)
        context = self.get_context(retrieved_docs)
        citations = self.get_citations(retrieved_docs)

        messages = [
            {
                "role": "system",
                "content": (
                    "Your name is Cloude (with an e at the end), you are a helpful AI assistant created by DefangLabs to help users learn about the cloud deployment tool Defang. "
                    "Your task is to provide positive answers about the cloud deployment tool Defang."
                    "When the user says 'you', 'your', or any pronoun, interpret it as referring to Ask Defang with context of Defang. "
                    "If the user's question involves comparisons with or references to other services, you may use external knowledge. "
                    "However, if the question is strictly about Defang, you must ignore all external knowledge and only utilize the given context. "
                    "Today's date is " + date.today().strftime("%B %d, %Y") + ". "
                    "Context: " + context
                ),
            }
        ]

        self.conversation_history.append({"role": "user", "content": query})
        messages.extend(self.conversation_history)

        try:
            logging.debug(f"Sending query to LLM: {normalized_query}")
            stream = openai.ChatCompletion.create(
                model=os.getenv("MODEL"),
                messages=messages,
                temperature=0.25,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True,
            )

            collected_messages = []
            for chunk in stream:
                try:
                    logging.debug(f"Received chunk: {chunk}")
                    content = chunk["choices"][0]["delta"].get("content", "")
                    collected_messages.append(content)
                    yield content
                    if chunk["choices"][0].get("finish_reason") is not None:
                        break
                except (BrokenPipeError, OSError) as e:
                    # Client disconnected, stop streaming
                    logging.warning(f"Client disconnected during streaming: {e}")
                    traceback.print_exc(file=sys.stderr)
                    break

            logging.debug(f"Finished receiving response: {normalized_query}")

            if len(citations) > 0:
                try:
                    yield "\n\nReferences:\n" + "\n".join(citations)
                except (BrokenPipeError, OSError) as e:
                    # Client disconnected, stop streaming
                    logging.warning(
                        f"Client disconnected during citations streaming: {e}"
                    )
                    traceback.print_exc(file=sys.stderr)

            full_response = "".join(collected_messages).strip()
            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            print(f"Error in answer_query_stream: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            try:
                yield "An error occurred while generating the response."
            except (BrokenPipeError, OSError):
                # Client disconnected, can't send error message
                logging.warning(
                    "Client disconnected before error message could be sent"
                )

    def clear_conversation_history(self):
        self.conversation_history = []
        print("Conversation history cleared.")

    def rebuild(self):
        """
        Rebuild the embeddings for the knowledge base. This should be called whenever the knowledge base is updated.
        """
        print("Rebuilding embeddings for the knowledge base...")
        knowledge_base = self.load_knowledge_base()  # Reload the knowledge base
        self.rebuild_embeddings(knowledge_base)  # Rebuild the embeddings
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    RAGSystem()
