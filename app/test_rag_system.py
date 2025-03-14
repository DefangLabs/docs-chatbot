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

    def test_calculate_relevance_scores(self):
        query_embedding = self.rag_system.model.encode(["peanut butter"], convert_to_tensor=True).cpu()
        similarities = [0.8, 0.6, 0.4] # instead of hardcoding these, get the real similarities
        relevance_scores = self.rag_system.calculate_relevance_scores(query_embedding, similarities, high_match_threshold=0.8)
        self.assertIsInstance(relevance_scores, list)
        self.assertGreater(len(relevance_scores), 0)
        print("Test for calculate_relevance_scores passed successfully!")

    def test_get_top_indices(self):
        relevance_scores = [(0, 0.9), (1, 0.7), (2, 0.5)]
        top_indices = self.rag_system.get_top_indices(relevance_scores, similarity_threshold=0.6, max_docs=2)
        self.assertIsInstance(top_indices, list)
        self.assertEqual(len(top_indices), 2)
        print("Test for get_top_indices passed successfully!")

    def test_get_top_docs(self):
        top_indices = [0, 2]
        top_docs = self.rag_system.get_top_docs(top_indices)
        self.assertIsInstance(top_docs, list)
        self.assertGreater(len(top_docs), 0)
        self.assertLessEqual(len(top_docs), 2)
        print("Test for get_top_docs passed successfully!")

    # def test_retrieve(self):
    #     query = "sample query"
    #     result = self.rag_system.retrieve(query)
    #     self.assertIsInstance(result, str)
    #     self.assertGreater(len(result), 0)
    #     print("Test for retrieve passed successfully!")

    def test_print_relevance_scores_matrix(self):
        # get embeddings and move to CPU
        query_embedding = self.rag_system.model.encode(["Samples"], convert_to_tensor=True).cpu()
        doc_embeddings = self.rag_system.doc_embeddings.cpu()

        # get the length of query embeddings
        self.assertEqual(len(query_embedding[0]), 384)

        # get the length of doc embeddings
        self.assertEqual(len(self.rag_system.doc_embeddings[0]), 384)
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)

        relevance_scores = self.rag_system.calculate_relevance_scores(query_embedding, similarities, high_match_threshold=0.8)
        for i, relevance_score in relevance_scores:
            print(f"{i}\t\t{relevance_score:.4f}")
        
        print("Relevance Scores Matrix:")
        print("Index\t\tSimilarity\tRelevance Score\t\tAbout")
        
        # print out similarities, abouts, and relevance scores
        for i, similarity in enumerate(similarities):
            about = self.rag_system.knowledge_base[i]["about"]
            relevance_score = relevance_scores[i][1]
            print(f"{i}\t\t{similarity.item():.4f}\t\t{relevance_score:.4f}\t\t\t{about}")
            
    #     # check if the answer gets back top 5 docs, wih their similarity scores

    # def test_upper(self):
    #         self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()