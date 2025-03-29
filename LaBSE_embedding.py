from sentence_transformers import SentenceTransformer


class LaBSEEmbedding:
    def __init__(self, model_name='sentence-transformers/LaBSE'):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
    
    def get_embedding(self, text):
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(text)
        return embeddings


if __name__ == "__main__":
    encoder = LaBSEEmbedding()
    