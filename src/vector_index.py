"""
Vector Index Builder for Device Specifications
Creates ChromaDB embeddings from device documents using Ollama
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")
    chromadb = None

try:
    import ollama
except ImportError:
    print("Ollama Python client not installed. Run: pip install ollama")
    ollama = None


class VectorIndexBuilder:
    """Builds and manages ChromaDB vector index for device documents."""
    
    def __init__(
        self,
        documents_path: str,
        persist_directory: str,
        collection_name: str = "device_specs",
        embedding_model: str = "nomic-embed-text"
    ):
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.documents = []
        self.client = None
        self.collection = None
        
    def load_documents(self) -> List[Dict]:
        """Load device documents from JSON file."""
        with open(self.documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"Loaded {len(self.documents)} documents from {self.documents_path}")
        return self.documents
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using Ollama."""
        if ollama is None:
            raise ImportError("Ollama client not installed")
        
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response['embedding']
    
    def initialize_database(self):
        """Initialize ChromaDB client and collection."""
        if chromadb is None:
            raise ImportError("ChromaDB not installed")
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Device specification documents for RAG"}
        )
        
        print(f"Initialized ChromaDB at {self.persist_directory}")
        print(f"Created collection: {self.collection_name}")
    
    def build_index(self, batch_size: int = 50) -> int:
        """Build vector index from all documents."""
        if not self.documents:
            self.load_documents()
        
        if self.collection is None:
            self.initialize_database()
        
        total = len(self.documents)
        indexed = 0
        
        print(f"\nIndexing {total} documents...")
        start_time = time.time()
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = self.documents[i:i + batch_size]
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for doc in batch:
                doc_id = doc.get('id') or f"doc_{indexed}"
                text = doc.get('text', '')
                
                if not text:
                    continue
                
                # Get embedding
                try:
                    embedding = self.get_embedding(text)
                except Exception as e:
                    print(f"Error embedding {doc_id}: {e}")
                    continue
                
                # Prepare metadata (ChromaDB only supports str, int, float, bool)
                metadata = {
                    'device_name': str(doc.get('device_name', '')),
                    'brand': str(doc.get('brand', '')),
                    'category': str(doc.get('category', '')),
                    'form_factor': str(doc.get('form_factor', '')),
                    'price_tier': str(doc.get('price_tier', '')),
                    'title': str(doc.get('title', '')),
                }
                
                # Add spec fields if available
                specs = doc.get('specs', {})
                if specs:
                    if specs.get('ram_gb'):
                        metadata['ram_gb'] = float(specs['ram_gb'])
                    if specs.get('storage_gb'):
                        metadata['storage_gb'] = float(specs['storage_gb'])
                    if specs.get('has_dedicated_gpu') is not None:
                        metadata['has_dedicated_gpu'] = bool(specs['has_dedicated_gpu'])
                    if specs.get('has_npu') is not None:
                        metadata['has_npu'] = bool(specs['has_npu'])
                    if specs.get('is_copilot_plus') is not None:
                        metadata['is_copilot_plus'] = bool(specs['is_copilot_plus'])
                    if specs.get('cpu_tier'):
                        metadata['cpu_tier'] = str(specs['cpu_tier'])
                    if specs.get('gpu_tier'):
                        metadata['gpu_tier'] = str(specs['gpu_tier'])
                
                ids.append(doc_id)
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append(metadata)
                indexed += 1
            
            # Add batch to collection
            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            
            # Progress update
            progress = min(i + batch_size, total)
            elapsed = time.time() - start_time
            rate = progress / elapsed if elapsed > 0 else 0
            print(f"  Progress: {progress}/{total} ({progress/total*100:.1f}%) - {rate:.1f} docs/sec")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Indexed {indexed} documents in {elapsed:.1f} seconds")
        
        return indexed
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents."""
        if self.collection is None:
            self.load_collection()
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Build search parameters
        search_params = {
            'query_embeddings': [query_embedding],
            'n_results': n_results,
            'include': ['documents', 'metadatas', 'distances']
        }
        
        if filter_dict:
            search_params['where'] = filter_dict
        
        # Execute search
        results = self.collection.query(**search_params)
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def load_collection(self):
        """Load existing collection from disk."""
        if chromadb is None:
            raise ImportError("ChromaDB not installed")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_collection(self.collection_name)
        print(f"Loaded collection '{self.collection_name}' with {self.collection.count()} documents")
    
    def get_device_by_name(self, device_name: str) -> Optional[Dict]:
        """Find a specific device by name (exact or partial match)."""
        if self.collection is None:
            self.load_collection()
        
        # First try metadata filter
        results = self.collection.get(
            where={"device_name": device_name},
            include=['documents', 'metadatas']
        )
        
        if results['ids']:
            return {
                'id': results['ids'][0],
                'document': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
        
        # Fall back to semantic search
        search_results = self.search(f"device {device_name}", n_results=1)
        if search_results and search_results[0]['similarity'] > 0.7:
            return search_results[0]
        
        return None
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.collection is None:
            self.load_collection()
        
        count = self.collection.count()
        
        # Sample to get metadata keys
        sample = self.collection.peek(limit=1)
        metadata_keys = list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
        
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'embedding_model': self.embedding_model,
            'metadata_fields': metadata_keys
        }


def main():
    """Main function to build vector index."""
    base_path = Path(__file__).parent.parent
    documents_path = base_path / 'data' / 'device_documents.json'
    persist_directory = base_path / 'data' / 'chromadb'
    
    if not documents_path.exists():
        print(f"Error: Documents not found at {documents_path}")
        print("Run run_phase1.py first to generate documents.")
        return
    
    # Build index
    builder = VectorIndexBuilder(
        documents_path=str(documents_path),
        persist_directory=str(persist_directory)
    )
    
    builder.load_documents()
    builder.initialize_database()
    builder.build_index()
    
    # Print stats
    stats = builder.get_stats()
    print(f"\n=== Index Statistics ===")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"Metadata fields: {stats['metadata_fields']}")
    
    # Test search
    print(f"\n=== Test Search ===")
    test_query = "gaming laptop with RTX GPU"
    results = builder.search(test_query, n_results=3)
    
    print(f"Query: '{test_query}'")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['metadata'].get('device_name', 'Unknown')} (similarity: {result['similarity']:.3f})")
        print(f"   Brand: {result['metadata'].get('brand')}, Category: {result['metadata'].get('category')}")


if __name__ == '__main__':
    main()
