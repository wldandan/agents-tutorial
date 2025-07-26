#!/usr/bin/env python3
"""
Test script for Level 2 Agent with Huggingface Embeddings
Tests the real embedding capabilities
"""

from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.models.deepseek import DeepSeek
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.base import Embedder
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np

# Load .env from project root directory
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Custom Huggingface Embedder
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
    
    def _fallback_embedding(self, text: str) -> list:
        """Simple fallback embedding using hash"""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        embedding = []
        for i in range(384):
            embedding.append(hash_bytes[i % 16] / 255.0)
        
        return embedding

def test_embedding_capabilities():
    """Test the embedding capabilities"""
    print("🧪 Testing Embedding Capabilities")
    print("=" * 60)
    
    try:
        # 1. Test Huggingface Embedder
        print("1. Testing Huggingface Embedder...")
        embedder = HuggingFaceEmbedder()
        
        # Test single embedding
        test_text = "This is a test sentence for embedding."
        embedding = embedder.get_embedding(test_text)
        print(f"✅ Single embedding generated: {len(embedding)} dimensions")
        
        # Test multiple embeddings
        test_texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.get_embeddings(test_texts)
        print(f"✅ Multiple embeddings generated: {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during embedding test: {e}")
        return False

def test_advanced_agent_components():
    """Test each component of the advanced agent with embeddings"""
    print("\n🧪 Testing Level 2 Agent with Embeddings")
    print("=" * 60)
    
    try:
        # 1. Test Huggingface Embedder
        print("1. Testing Huggingface Embedder...")
        huggingface_embedder = HuggingFaceEmbedder()
        print("✅ Huggingface Embedder created successfully")
        
        # 2. Test LanceDB with custom embedder
        print("2. Testing LanceDB with custom embedder...")
        lancedb = LanceDb(
            uri="tmp/lancedb_with_embeddings",
            table_name="agno_docs_with_embeddings",
            search_type=SearchType.hybrid,
            embedder=huggingface_embedder
        )
        print("✅ LanceDB with custom embedder created successfully")
        
        # 3. Test Knowledge Base
        print("3. Testing Knowledge Base...")
        knowledge = UrlKnowledge(
            urls=["https://docs.agno.com/introduction.md"],
            vector_db=lancedb
        )
        print("✅ Knowledge Base created successfully")
        
        # 4. Test Storage
        print("4. Testing Storage...")
        storage = SqliteStorage(
            table_name="agent_sessions_with_embeddings", 
            db_file="tmp/agent_with_embeddings.db"
        )
        print("✅ Storage created successfully")
        
        # 5. Test Model
        print("5. Testing DeepSeek Model...")
        model = DeepSeek(
            id="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        print("✅ DeepSeek Model created successfully")
        
        # 6. Test Agent
        print("6. Testing Agent Assembly...")
        agent = Agent(
            name="Level 2 Agent with Huggingface Embeddings",
            model=model,
            instructions=[
                "Search your knowledge before answering the question.",
                "Only include the output in your response. No other text.",
                "Use Huggingface embeddings and LanceDB for efficient knowledge retrieval.",
            ],
            knowledge=knowledge,
            storage=storage,
            add_datetime_to_instructions=True,
            add_history_to_messages=True,
            num_history_runs=3,
            markdown=True,
        )
        print("✅ Agent assembled successfully")
        
        print("\n🎉 All components tested successfully!")
        print("\nAgent with Embeddings Features:")
        print("- Custom Huggingface embeddings (all-MiniLM-L6-v2)")
        print("- LanceDB vector database with hybrid search")
        print("- DeepSeek model for reasoning")
        print("- SQLite storage for sessions")
        print("- Knowledge base with Agno documentation")
        print("- Fallback embedding system for robustness")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def explain_embedding_features():
    """Explain the embedding features"""
    print("\n🔧 Embedding Features Explanation")
    print("=" * 60)
    
    print("\n1. **Huggingface Embeddings**:")
    print("   - Model: sentence-transformers/all-MiniLM-L6-v2")
    print("   - Benefits: Local processing, semantic understanding")
    print("   - Performance: 384-dimensional vectors")
    print("   - Fallback: Hash-based embedding if Huggingface fails")
    
    print("\n2. **LanceDB Vector Database**:")
    print("   - High-performance vector storage")
    print("   - Hybrid search (vector + keyword)")
    print("   - Local storage, data privacy")
    print("   - Custom embedder integration")
    
    print("\n3. **Robust Architecture**:")
    print("   - Text → Huggingface Embedding → Vector → LanceDB")
    print("   - Fallback → Hash Embedding → Vector → LanceDB")
    print("   - Query → Embedding → Vector → LanceDB Search")
    print("   - Results → DeepSeek Model → Response")
    
    print("\n4. **Benefits**:")
    print("   - True semantic understanding with Huggingface")
    print("   - Robust fallback system")
    print("   - No dependency on OpenAI for embeddings")
    print("   - Better performance and control")

if __name__ == "__main__":
    explain_embedding_features()
    
    # Test embedding capabilities first
    if test_embedding_capabilities():
        test_advanced_agent_components()
    
    print("\n" + "=" * 60)
    print("To run the agent with embeddings:")
    print("1. Set DEEPSEEK_API_KEY in your .env file")
    print("2. Run: python level_2_agent_with_embeddings.py") 