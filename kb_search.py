from pathlib import Path

from dotenv import load_dotenv
import chromadb

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class SemanticSearch:
    
    def __init__(self) -> None:

        load_dotenv()  
        base_dir = Path(__file__).resolve().parent # Absolute anchoring based on file's position
        self.kb_path = base_dir / "knowledge_base"
        self.db_path = base_dir / "knowledge_base" / "vector_database"

        self.allowed_filetype = [".pdf", ".docx", ".txt", ".md", ".html", ".csv"]
        self.client = chromadb.PersistentClient(path=self.db_path)
        embedding_function = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_vectors",
            metadata={"hnsw:space": "cosine"},
            embedding_function = embedding_function)
        
