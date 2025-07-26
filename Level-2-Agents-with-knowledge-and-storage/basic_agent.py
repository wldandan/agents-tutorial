from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.models.deepseek import DeepSeek
from agno.storage.sqlite import SqliteStorage
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root directory
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Load Agno documentation in a knowledge base
# You can also use `https://docs.agno.com/llms-full.txt` for the full documentation
knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/introduction.md"],
)

# Store agent sessions in a SQLite database
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")

agent = Agent(
    name="Agno Assist",
    model=DeepSeek(
        id="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    ),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
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