from src.dwarf.core.utils.utils import *

class EmbeddingCore():
    def __init__(self):
        self.available_embeddings: dict = {}
        pass

    def get_new_embedding(self, embeddings: dict):

        for embedding in embeddings:
            # self.available_embeddings[embedding[embedding.rfind("\\", 0, len(embedding)) + 1:]] = import_function()
            self.available_embeddings[embedding] = import_function(embedding, embeddings[embedding])
        return
    
    def get_all_available_embeddings(self):
        for embedding in self.available_embeddings:
            print(embedding)
        return