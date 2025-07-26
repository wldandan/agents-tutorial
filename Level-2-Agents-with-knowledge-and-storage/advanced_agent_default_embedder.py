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

# 1. LanceDB - 高性能向量数据库（使用默认 embedder）
lancedb = LanceDb(
    uri="tmp/lancedb_advanced_simple",        # 数据库路径
    table_name="agno_docs_advanced_simple",   # 表名
    search_type=SearchType.hybrid,            # 混合搜索（向量+关键词）
)

# 2. Knowledge - 使用 LanceDB 存储知识
knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/introduction.md"],
    vector_db=lancedb  # 指定使用 LanceDB
)

# 3. Storage - 会话存储
storage = SqliteStorage(
    table_name="agent_sessions_advanced_simple", 
    db_file="tmp/agent_advanced_simple.db"
)

# 4. Agent - 组合所有组件
agent = Agent(
    name="Level 2 Advanced Simple Agent",
    model=DeepSeek(
        id="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    ),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
        "Use LanceDB for efficient knowledge retrieval.",
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
    # Load the knowledge base, comment out after first run
    # Set recreate to True to recreate the knowledge base if needed
    agent.knowledge.load(recreate=False)
    agent.print_response("What is Agno?", stream=True) 