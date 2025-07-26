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

# 1. Custom Huggingface Embedder - 真正的 Huggingface 嵌入器
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
            # Fallback: simple hash-based embedding
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
        # Simple usage info - in real implementation you might track tokens
        usage = {"tokens": len(text.split()), "model": self.model_name}
        return embedding, usage
    
    def _fallback_embedding(self, text: str) -> list:
        """Simple fallback embedding using hash"""
        import hashlib
        # Create a simple 384-dimensional embedding using hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 384-dimensional vector
        embedding = []
        for i in range(384):
            embedding.append(hash_bytes[i % 16] / 255.0)
        
        return embedding

# 2. 创建 embedder 实例
print("🔧 Initializing Huggingface Embedder...")
huggingface_embedder = HuggingFaceEmbedder()

# 3. LanceDB - 高性能向量数据库
print("🔧 Setting up LanceDB...")
lancedb = LanceDb(
    uri="tmp/lancedb_with_embeddings",        # 数据库路径
    table_name="agno_docs_with_embeddings",   # 表名
    search_type=SearchType.hybrid,            # 混合搜索（向量+关键词）
    embedder=huggingface_embedder             # 使用自定义 Huggingface embedder
)

# 4. Knowledge - 使用 LanceDB 存储知识
print("🔧 Setting up Knowledge Base...")
knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/introduction.md"],
    vector_db=lancedb  # 指定使用 LanceDB
)

# 5. Storage - 会话存储
print("🔧 Setting up Storage...")
storage = SqliteStorage(
    table_name="agent_sessions_with_embeddings", 
    db_file="tmp/agent_with_embeddings.db"
)

# 6. Agent - 组合所有组件
print("🔧 Creating Agent...")
agent = Agent(
    name="Level 2 Agent with Huggingface Embeddings",
    model=DeepSeek(
        id="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    ),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
        "Use Huggingface embeddings and LanceDB for efficient knowledge retrieval.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_datetime_to_instructions=True,
    # Add the chat history to the messages
    add_history_to_messages=True,
    # Number of history runs
    num_history_runs=3,
    markdown=True,
)

if __name__ == "__main__":
    print("🚀 Starting Level 2 Agent with Huggingface Embeddings...")
    
    # Load the knowledge base, comment out after first run
    # Set recreate to True to recreate the knowledge base if needed
    print("📚 Loading knowledge base...")
    agent.knowledge.load(recreate=False)
    
    print("💬 Asking question...")
    agent.print_response("What is Agno?", stream=True) 