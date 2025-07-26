#!/usr/bin/env python3
"""
Test script for Level 2 Advanced Simple Agent
Tests LanceDB configuration without complex dependencies
"""

from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.models.deepseek import DeepSeek
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root directory
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

def test_advanced_simple_agent():
    """Test the simplified advanced agent"""
    print("🧪 Testing Level 2 Advanced Simple Agent")
    print("=" * 60)
    
    try:
        # 1. Test LanceDB
        print("1. Testing LanceDB...")
        lancedb = LanceDb(
            uri="tmp/lancedb_advanced_simple",
            table_name="agno_docs_advanced_simple",
            search_type=SearchType.hybrid,
        )
        print("✅ LanceDB created successfully")
        
        # 2. Test Knowledge Base
        print("2. Testing Knowledge Base...")
        knowledge = UrlKnowledge(
            urls=["https://docs.agno.com/introduction.md"],
            vector_db=lancedb
        )
        print("✅ Knowledge Base created successfully")
        
        # 3. Test Storage
        print("3. Testing Storage...")
        storage = SqliteStorage(
            table_name="agent_sessions_advanced_simple", 
            db_file="tmp/agent_advanced_simple.db"
        )
        print("✅ Storage created successfully")
        
        # 4. Test Model
        print("4. Testing DeepSeek Model...")
        model = DeepSeek(
            id="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        print("✅ DeepSeek Model created successfully")
        
        # 5. Test Agent
        print("5. Testing Agent Assembly...")
        agent = Agent(
            name="Level 2 Advanced Simple Agent",
            model=model,
            instructions=[
                "Search your knowledge before answering the question.",
                "Only include the output in your response. No other text.",
                "Use LanceDB for efficient knowledge retrieval.",
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
        print("\nAdvanced Simple Agent Features:")
        print("- LanceDB vector database with hybrid search")
        print("- DeepSeek model for reasoning")
        print("- SQLite storage for sessions")
        print("- Knowledge base with Agno documentation")
        print("- Default embedder (no complex dependencies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def explain_simple_advanced_features():
    """Explain the simplified advanced features"""
    print("\n🔧 Simplified Advanced Features")
    print("=" * 60)
    
    print("\n1. **LanceDB Vector Database**:")
    print("   - High-performance vector storage")
    print("   - Hybrid search (vector + keyword)")
    print("   - Local storage, data privacy")
    print("   - Scalable for large datasets")
    
    print("\n2. **Default Embedder**:")
    print("   - Uses Agno's default embedding system")
    print("   - No complex dependencies required")
    print("   - Compatible with OpenAI embeddings")
    
    print("\n3. **Architecture**:")
    print("   - Text → Default Embedding → Vector → LanceDB")
    print("   - Query → Default Embedding → Vector → LanceDB Search")
    print("   - Results → DeepSeek Model → Response")
    
    print("\n4. **Benefits**:")
    print("   - Easy to set up and run")
    print("   - Better performance than basic version")
    print("   - No dependency conflicts")
    print("   - Production-ready vector database")

if __name__ == "__main__":
    explain_simple_advanced_features()
    test_advanced_simple_agent()
    
    print("\n" + "=" * 60)
    print("To run the simplified advanced agent:")
    print("1. Set DEEPSEEK_API_KEY in your .env file")
    print("2. Run: python level_2_agent_advanced_simple.py") 