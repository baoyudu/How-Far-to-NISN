from sentence_transformers import SentenceTransformer
import os

class JinaEmbedding:
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
    
    def get_embedding(self, text, task="retrieval.query"):
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(
            text,
            task=task,
            prompt_name=task,
        )
        return embeddings


if __name__ == "__main__":
    encoder = JinaEmbedding()

    # import time
    
    # text = "What is the weather like in Berlin today?"
    # start_time = time.time()
    # embedding = encoder.get_embedding(text)
    # end_time = time.time()
    
    # print(f"Single text embedding shape: {embedding.shape}")
    # print(f"Single text embedding: {embedding}")
    # print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    # texts = ["Hello world!", "How are you?", "Nice to meet you!"]
    # embeddings = encoder.get_embedding(texts)
    # print(f"Multiple texts embedding shape: {embeddings.shape}")