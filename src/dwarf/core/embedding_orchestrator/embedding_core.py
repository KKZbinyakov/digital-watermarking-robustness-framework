from src.dwarf.core.utils.utils import *

if TYPE_CHECKING:
    from src.dwarf.ready_solutions.embedding_solutions.spatial_embeddings import Ready_Spatial_Embeddings

class Embedding_Core():
    if TYPE_CHECKING:
        spatial_embeddings: Ready_Spatial_Embeddings
    def __init__(self):
        self.available_embeddings: dict = {}
        self.ready_embedding_types: dict = {
            "spatial_embeddings": str(DWARF_DIR) + r"\ready_solutions\embedding_solutions\spatial_embeddings.py",
        }
        pass

    def __getattr__(self, embedding_type: str):
        if embedding_type in self.available_embeddings:
            return self.available_embeddings[embedding_type]
        else:
            raise AttributeError

    def get_new_embedding(self, embeddings: dict):

        for embedding in embeddings:
            # self.available_embeddings[embedding[embedding.rfind("\\", 0, len(embedding)) + 1:]] = import_function()
            self.available_embeddings[embedding] = import_function(embedding, embeddings[embedding])
        return
    
    def get_all_available_embeddings(self):
        for embedding in self.available_embeddings:
            print(embedding)
        return
    
    def connect_ready_embeddings(self, embeddings_type: list = [], connect_all: bool = False):
        for embedding in embeddings_type:
            self.available_embeddings[embedding] = import_function(embedding, self.ready_embedding_types[embedding])

        if connect_all:
            for embedding in self.ready_embedding_types:
                self.available_embeddings[embedding] = import_function(embedding, self.ready_embedding_types[embedding])
        return