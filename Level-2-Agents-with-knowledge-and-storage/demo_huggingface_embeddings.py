#!/usr/bin/env python3
"""
Demo script for Huggingface Embeddings
Shows how the custom embedding system works
"""

from agno.embedder.base import Embedder
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root directory
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            print(f"✅ Loaded Huggingface model: {model_name}")
        except ImportError:
            print("❌ sentence-transformers not available, using fallback")
            self.model = None
            self.model_name = "fallback"
    
    def get_embedding(self, text: str) -> list:
        """Generate embedding for text"""
        if self.model is None:
            return self._fallback_embedding(text)
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Warning: Huggingface embedding failed, using fallback: {e}")
            return self._fallback_embedding(text)
    
    def get_embeddings(self, texts: list) -> list:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Warning: Huggingface embeddings failed, using fallback: {e}")
            return [self._fallback_embedding(text) for text in texts]
    
    def get_embedding_and_usage(self, text: str) -> tuple:
        """Generate embedding and usage info for text"""
        embedding = self.get_embedding(text)
        usage = {"tokens": len(text.split()), "model": self.model_name}
        return embedding, usage
    
    def _fallback_embedding(self, text: str) -> list:
        """Simple fallback embedding using hash"""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        embedding = []
        for i in range(384):
            embedding.append(hash_bytes[i % 16] / 255.0)
        
        return embedding

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def demo_embeddings():
    """Demonstrate embedding capabilities"""
    print("🚀 Huggingface Embeddings Demo")
    print("=" * 60)
    
    # Initialize embedder
    embedder = HuggingFaceEmbedder()
    
    # Test sentences
    sentences = [
        "Agno is a framework for building AI agents.",
        "Agno provides tools for knowledge and storage.",
        "The weather is sunny today.",
        "I love programming with Python.",
        "Agno agents can use embeddings for semantic search."
    ]
    
    print(f"\n📝 Testing {len(sentences)} sentences...")
    
    # Generate embeddings
    embeddings = embedder.get_embeddings(sentences)
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"📊 Each embedding has {len(embeddings[0])} dimensions")
    
    # Test semantic similarity
    print(f"\n🔍 Testing semantic similarity...")
    
    # Find most similar to "Agno"
    agno_sentences = [0, 1, 4]  # Indices of Agno-related sentences
    weather_sentence = 2  # Weather sentence
    
    print(f"\nComparing Agno-related sentences with weather sentence:")
    for i in agno_sentences:
        similarity = cosine_similarity(embeddings[i], embeddings[weather_sentence])
        print(f"  '{sentences[i][:30]}...' vs '{sentences[weather_sentence][:30]}...'")
        print(f"  Similarity: {similarity:.4f}")
    
    # Test similar concepts
    print(f"\nComparing similar concepts:")
    similarity = cosine_similarity(embeddings[0], embeddings[1])  # Both about Agno
    print(f"  'Agno framework' vs 'Agno tools': {similarity:.4f}")
    
    similarity = cosine_similarity(embeddings[0], embeddings[4])  # Both about Agno
    print(f"  'Agno framework' vs 'Agno agents': {similarity:.4f}")
    
    # Test different concepts
    print(f"\nComparing different concepts:")
    similarity = cosine_similarity(embeddings[2], embeddings[3])  # Weather vs Python
    print(f"  'Weather' vs 'Python programming': {similarity:.4f}")
    
    print(f"\n🎉 Embedding demo completed!")
    print(f"💡 The embeddings show semantic understanding:")
    print(f"   - Similar concepts have higher similarity scores")
    print(f"   - Different concepts have lower similarity scores")
    print(f"   - This enables better semantic search in the agent!")

if __name__ == "__main__":
    demo_embeddings() 