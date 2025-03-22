import unittest

from torch import cosine_similarity
from rag_system import RAGSystem

class TestRAGSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rag_system = RAGSystem(knowledge_base_path='test_knowledge_base.json')
        cls.rag_system.rebuild_embeddings()
        cls.initial_embeddings = cls.rag_system.doc_embeddings.clone()
        assert cls.initial_embeddings is not None, "Embeddings were not rebuilt properly."
        print("Successfully set up RAG System class for testing!")

    def test_normalize_query(self):
        query = "  Hello World  "
        normalized_query = self.rag_system.normalize_query(query)
        self.assertEqual(normalized_query, "hello world")
        print("Test for normalize_query passed successfully!")

    def test_get_top_docs(self):
        doc_scores = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.6},
            {"index": 2, "relevance_score": 0.7}
        ]
        top_docs = self.rag_system.get_top_docs(doc_scores, similarity_threshold=0.7, max_docs=2)
        self.assertIsInstance(top_docs, list)
        self.assertGreater(len(top_docs), 0)
        self.assertLessEqual(len(top_docs), 2)
        for doc in top_docs:
            self.assertIn("index", doc)
            self.assertIn(doc["index"], [0, 2]) # should have indices where relevance scores >= similarity threshold
            self.assertIn("relevance_score", doc)
        print("Test for get_top_docs passed successfully!")

    def test_get_query_embedding(self):
        query = "What is Defang?"
        query_embedding = self.rag_system.get_query_embedding(query)
        self.assertIsNotNone(query_embedding)
        self.assertEqual(len(query_embedding.shape), 2)  # should be a 2D tensor
        self.assertEqual(query_embedding.shape[0], 1) # should have exactly one embedding
        print("Test for get_query_embedding passed successfully!")

    def test_get_doc_embeddings(self):
        doc_embeddings = self.rag_system.get_doc_embeddings()
        self.assertIsNotNone(doc_embeddings)
        self.assertEqual(len(doc_embeddings.shape), 2)  # should be a 2D tensor
        self.assertGreater(doc_embeddings.shape[0], 0)  # should have at least one document embedding
        print("Test for get_doc_embeddings passed successfully!")

    def test_retrieve_fallback(self):
        # test a query that should return the fallback response
        query = "Hello"
        # set use_cpu to True, as testing has no GPU calculations
        result = self.rag_system.retrieve(query, use_cpu=True)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(len(result), 1)  # should return one result for fallback
        for doc in result:
            self.assertIn("about", doc)
            self.assertIn("text", doc)
            self.assertEqual(doc['about'], "No Relevant Information Found")
        print("Test for retrieve_fallback passed successfully!")

    def test_retrieve_actual_response(self):
        # test a query that should return an actual response from the knowledge base
        query = "What is Defang?"
        # set use_cpu to True, as testing has no GPU calculations
        result = self.rag_system.retrieve(query, use_cpu=True)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 5)  # should return up to max_docs (5)
        for doc in result:
            self.assertIn("about", doc)
            self.assertIn("text", doc)
            self.assertNotEqual(doc['about'], "No Relevant Information Found")
        print("Test for retrieve_actual_response passed successfully!")

    def test_compute_document_scores(self):
        query = "Does Defang have an MCP sample?"
        # get embeddings and move them to CPU, as testing has no GPU calculations
        query_embedding = self.rag_system.get_query_embedding(query, use_cpu=True)
        doc_embeddings = self.rag_system.get_doc_embeddings(use_cpu=True)

        # call function and get results
        result = self.rag_system.compute_document_scores(query_embedding, doc_embeddings, high_match_threshold=0.8)
        # sort the result by relevance score in descending order
        result = sorted(result, key=lambda x: x["relevance_score"], reverse=True)

        # print the results
        print("Index\tText Sim.\tAbout Sim.\tRelevance Score\tAbout")
        for doc in result:
            about = self.rag_system.knowledge_base[doc["index"]]["about"]
            if len(about) > 50:  # cut off if 'about' is too long
                about = about[:47] + "..."
            # print the doc scores
            print(f"{doc['index']}\t" + "\t\t".join(f"{score:.4f}" for score in [
            doc["text_similarity"],
            doc["about_similarity"],
            doc["relevance_score"]
            ]) + f"\t\t{about}")

        print("Test for compute_document_scores passed successfully!")

if __name__ == '__main__':
    unittest.main()