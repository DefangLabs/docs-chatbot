import os
from sentence_transformers import SentenceTransformer

def load_model():
    """
    Load the SentenceTransformer model from the specified path.
    The model path is determined by the SENTENCE_TRANSFORMERS_HOME environment variable.
    """
    # model_path = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./models/sentence-transformers")
    # model = SentenceTransformer(f"{model_path}/models--sentence-transformers--all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

if __name__ == "__main__":
    load_model()
