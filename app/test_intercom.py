import time
import unittest
from unittest.mock import patch, Mock
import fakeredis

# Apply patch to use a fake redis for testing before importing app
redis_mock = fakeredis.FakeStrictRedis(decode_responses=True)
patch('redis.from_url', return_value=redis_mock).start()

# Now import app, after the patch is applied
import intercom

class TestIntercom(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the app or any required resources
        cls.app = intercom
        # Replace the Redis client in app with our mock
        intercom.r = redis_mock
        print("Successfully set up App class for testing!")

    def test_parse_html_to_text(self):
        self.assertEqual(self.app.parse_html_to_text("<p>Hello, world!</p>"), "Hello, world!")
        print("test_parse_html_to_text passed successfully.")

    def test_parse_html_to_text_multiple_tags(self):
        html = "<div>Hello <b>world</b>!<br>How are <i>you</i>?</div>"
        result = self.app.parse_html_to_text(html)
        self.assertEqual(result, "Hello world!How are you?")
        print("test_parse_html_to_text_multiple_tags passed successfully.")

    def test_extract_conversation_parts(self):
        # Simulate a response object with .json() method
        mock_response = Mock()
        mock_response.json.return_value = {
            'conversation_parts': {
                'conversation_parts': [
                    {
                    'body': '<p>User message</p>',
                    'author': {'type': 'user'},
                    'created_at': 123,
                    'extra_field': 'foo'
                    },
                    {
                    'body': '<p>Admin reply</p>',
                    'author': {'type': 'admin'},
                    'created_at': 124,
                    'extra_field': 'bar'
                    },
                    {
                    'body': '',
                    'author': {'type': 'user'},
                    'created_at': 125,
                    'extra_field': 'baz'
                    }
                ]
            }
        }
        result = self.app.extract_conversation_parts(mock_response)
        expected = [
            {'body': '<p>User message</p>', 'author': {'type': 'user'}, 'created_at': 123},
            {'body': '<p>Admin reply</p>', 'author': {'type': 'admin'}, 'created_at': 124}
        ]
        self.assertEqual(result, expected)
        print("test_extract_conversation_parts passed successfully.")

    def test_extract_latest_user_messages_empty(self):
        conversation_parts = []
        result = self.app.extract_latest_user_messages(conversation_parts)
        self.assertIsNone(result)
        print("test_extract_latest_user_messages_empty passed successfully.")

    def test_extract_latest_user_messages(self):
        # Simulate conversation parts with user and admin messages
        conversation_parts = [
            {'body': '<p>Admin message</p>', 'author': {'type': 'admin'}, 'created_at': 1},
            {'body': '<p>User message 1</p>', 'author': {'type': 'user'}, 'created_at': 2},
            {'body': '<p>User message 2</p>', 'author': {'type': 'user'}, 'created_at': 3},
        ]
        result = self.app.extract_latest_user_messages(conversation_parts)
        self.assertEqual(result, "User message 1 User message 2")
        print("test_extract_latest_user_messages passed successfully.")

    def test_extract_latest_user_messages_no_user(self):
        conversation_parts = [
            {'body': '<p>Admin message</p>', 'author': {'type': 'admin'}, 'created_at': 1}
        ]
        result = self.app.extract_latest_user_messages(conversation_parts)
        self.assertIsNone(result)
        print("test_extract_latest_user_messages_no_user passed successfully.")

    def test_extract_latest_user_messages_all_users(self):
        conversation_parts = [
            {'body': '<p>User message 1</p>', 'author': {'type': 'user'}, 'created_at': 1},
            {'body': '<p>User message 2</p>', 'author': {'type': 'user'}, 'created_at': 2},
        ]
        result = self.app.extract_latest_user_messages(conversation_parts)
        self.assertEqual(result, "User message 1 User message 2")
        print("test_extract_latest_user_messages_all_users passed successfully.")

    def test_extract_latest_user_messages_after_admin(self):
        conversation_parts = [
            {'body': '<p>Admin message</p>', 'author': {'type': 'admin'}, 'created_at': 1},
            {'body': '<p>User message 1</p>', 'author': {'type': 'user'}, 'created_at': 2},
            {'body': '<p>User message 2</p>', 'author': {'type': 'user'}, 'created_at': 3},
            {'body': '<p>Admin message 2</p>', 'author': {'type': 'admin'}, 'created_at': 4},
            {'body': '<p>User message 3</p>', 'author': {'type': 'user'}, 'created_at': 5},
        ]
        result = self.app.extract_latest_user_messages(conversation_parts)
        self.assertEqual(result, "User message 3")
        print("test_extract_latest_user_messages_after_admin passed successfully.")

    def test_is_conversation_human_replied_check_false(self):
        conversation_id = "test_convo_id_1234"
        self.app.r.delete(conversation_id)
        result = self.app.is_conversation_human_replied(conversation_id, redis_mock)
        self.assertFalse(result)
        print("test_is_conversation_human_replied_check_false passed successfully.")

    def test_set_conversation_human_replied_and_check_true(self):
        conversation_id = "test_convo_id_1235"
        self.app.r.delete(conversation_id)
        self.app.set_conversation_human_replied(conversation_id, redis_mock)
        result = self.app.is_conversation_human_replied(conversation_id, redis_mock)
        self.assertTrue(result)
        self.app.r.delete(conversation_id)
        print("test_set_conversation_human_replied_and_check_true passed successfully.")

    def test_set_conversation_human_replied_ttl_exists(self):
        conversation_id = "test_convo_id_ttl"
        self.app.r.delete(conversation_id)
        self.app.set_conversation_human_replied(conversation_id, redis_mock)
        ttl = self.app.r.ttl(conversation_id)
        self.assertGreater(ttl, 0)
        self.app.r.delete(conversation_id)
        print("test_set_conversation_human_replied_ttl_exists passed successfully.")

    def test_set_redis_ttl_expiry(self):
        conversation_id = "test_convo_id_expiry"
        self.app.r.delete(conversation_id)
        # Set with a short TTL for test
        self.app.r.set(conversation_id, '1', ex=1)
        time.sleep(2)
        exists = self.app.r.exists(conversation_id)
        self.assertFalse(exists)
        print("test_set_redis_ttl_expiry passed successfully.")

    def test_check_intercom_ip_allowed(self):
        # Simulate a request with an allowed IP in X-Forwarded-For
        class DummyRequest:
            headers = {'X-Forwarded-For': '34.231.68.152'}
            remote_addr = '1.2.3.4'
        req = DummyRequest()
        result = self.app.check_intercom_ip(req)
        self.assertTrue(result)
        print("test_check_intercom_ip_allowed passed successfully.")

    def test_check_intercom_ip_allowed_remote_addr(self):
        # Simulate a request with allowed IP only in remote_addr
        class DummyRequest:
            headers = {}
            remote_addr = '34.197.76.213'
        req = DummyRequest()
        result = self.app.check_intercom_ip(req)
        self.assertTrue(result)
        print("test_check_intercom_ip_allowed_remote_addr passed successfully.")

    def test_check_intercom_ip_not_allowed(self):
        # Simulate a request with a non-allowed IP
        class DummyRequest:
            headers = {'X-Forwarded-For': '8.8.8.8'}
            remote_addr = '8.8.4.4'
        req = DummyRequest()
        result = self.app.check_intercom_ip(req)
        self.assertFalse(result)
        print("test_check_intercom_ip_not_allowed passed successfully.")

    def test_check_intercom_ip_multiple_forwarded(self):
        # Simulate a request with multiple IPs in X-Forwarded-For
        class DummyRequest:
            headers = {'X-Forwarded-For': '8.8.8.8, 34.231.68.152'}
            remote_addr = '8.8.4.4'
        req = DummyRequest()
        result = self.app.check_intercom_ip(req)
        self.assertFalse(result)
        print("test_check_intercom_ip_multiple_forwarded passed successfully.")

    def test_check_intercom_ip_none(self):
        # Simulate a request with no IPs
        class DummyRequest:
            headers = {}
            remote_addr = None
        req = DummyRequest()
        result = self.app.check_intercom_ip(req)
        self.assertFalse(result)
        print("test_check_intercom_ip_none passed successfully.")

if __name__ == '__main__':
    unittest.main()
