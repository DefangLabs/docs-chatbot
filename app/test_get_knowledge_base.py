import unittest
from get_knowledge_base import adjust_knowledge_base_entry_path

class TestGetKnowledgeBase(unittest.TestCase):

    def test_adjust_knowledgebase_entry_path(self):
        file_path = "./.tmp/defang-docs/2023-03-15-sample.md"
        adjusted_path = adjust_knowledge_base_entry_path(file_path)
        self.assertEqual(adjusted_path, "/2023/03/15/sample")
        print("Test for adjust_knowledgebase_entry_path passed successfully!")

    def test_adjust_knowledgebase_entry_path_no_date(self):
        file_path = "./.tmp/defang-docs/sample.mdx"
        adjusted_path = adjust_knowledge_base_entry_path(file_path)
        self.assertEqual(adjusted_path, "/sample")
        print("Test for adjust_knowledgebase_entry_path_no_date passed successfully!")

    def test_adjust_knowledgebase_entry_path_no_extension(self):
        file_path = "./.tmp/defang-docs/sample"
        adjusted_path = adjust_knowledge_base_entry_path(file_path)
        self.assertEqual(adjusted_path, "/sample")
        print("Test for adjust_knowledgebase_entry_path_no_extension passed successfully!")

    def test_adjust_knowledgebase_entry_path_empty(self):
        file_path = ""
        adjusted_path = adjust_knowledge_base_entry_path(file_path)
        self.assertEqual(adjusted_path, "")
        print("Test for adjust_knowledgebase_entry_path_empty passed successfully!")

    def test_adjust_knowledgebase_entry_path_date_at_end(self):
        file_path = "./.tmp/defang-docs/example-file-2023-03-15.md"
        adjusted_path = adjust_knowledge_base_entry_path(file_path)
        self.assertEqual(adjusted_path, "/example-file-2023-03-15")
        print("Test for adjust_knowledgebase_entry_path_date_at_end passed successfully!")

    def test_adjust_knowledgebase_entry_path_date_at_middle(self):
        file_path = "./.tmp/defang-docs/example-file-2023-03-15-something-else.md"
        adjusted_path = adjust_knowledge_base_entry_path(file_path)
        self.assertEqual(adjusted_path, "/example-file-2023-03-15-something-else")
        print("Test for adjust_knowledgebase_entry_path_date_at_middle passed successfully!")


if __name__ == '__main__':
    unittest.main()
