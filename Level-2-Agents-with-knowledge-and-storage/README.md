# Level 2: Agent with Knowledge and Storage

This project demonstrates a Level 2 Agent using Agno framework with knowledge and storage capabilities, following the [official Agno documentation](https://docs.agno.com/introduction/agents#level-2%3A-agents-with-knowledge-and-storage).

## Features

- **Knowledge Base**: Uses LanceDB with OpenAI embeddings to store and search Agno documentation
- **Storage**: SQLite database for persistent session history and state
- **Model**: Claude Sonnet 4 for reasoning and response generation
- **Hybrid Search**: Combines vector and keyword search for better results

## Architecture

### Components

1. **Knowledge**: `UrlKnowledge` loads Agno documentation into LanceDB
2. **Storage**: `SqliteStorage` saves agent sessions and chat history
3. **Agent**: Combines knowledge, storage, and Claude model for intelligent responses

### Technology Stack

- **Framework**: Agno
- **Embeddings**: Default Agno embeddings (OpenAI compatible)
- **Vector Database**: Built-in Agno vector storage
- **Storage**: SQLite
- **Model**: DeepSeek Chat

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:

Create a `.env` file in the project root directory (`/Users/leiw/Projects/agents-tutorial/.env`):
```bash
# DeepSeek API Key
DEEPSEEK_API_KEY=sk-***

# OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-***
```

Or set environment variables directly:
```bash
export DEEPSEEK_API_KEY=sk-***
```

## Usage

### Testing

First, test the agent initialization:
```bash
python test_agent.py
```

### Basic Usage

Set your DeepSeek API key:
```bash
export DEEPSEEK_API_KEY=sk-***
```

Run the basic agent:
```bash
python basic_agent.py
```

The agent will:
1. Load Agno documentation into the knowledge base (first run only)
2. Ask "What is Agno?" and provide a response
3. Save the conversation to SQLite storage

### Advanced Usage

The advanced agent uses LanceDB for enhanced vector storage:

```bash
# Test advanced agent with default embedder
python test_advanced_agent_default_embedder.py

# Run the advanced agent with default embedder
python advanced_agent_default_embedder.py
```

Advanced features:
- **LanceDB**: High-performance vector database with hybrid search
- **Default Embeddings**: Uses Agno's default embedding system
- **Enhanced Performance**: Better scalability and control
- **No Complex Dependencies**: Easy to set up and run

### Advanced Usage with Huggingface Embeddings

The agent with embeddings uses custom Huggingface embeddings for better semantic understanding:

```bash
# Test advanced agent with Huggingface embeddings
python test_advanced_agent_huggingface_embeddings.py

# Run the advanced agent with Huggingface embeddings
python advanced_agent_huggingface_embeddings.py

# Demo Huggingface embeddings functionality
python demo_huggingface_embeddings.py
```

Embeddings features:
- **Huggingface Embeddings**: Custom sentence-transformers/all-MiniLM-L6-v2 model
- **LanceDB**: High-performance vector database with hybrid search
- **Semantic Understanding**: Better text similarity and search
- **Fallback System**: Robust hash-based embedding if Huggingface fails
- **Local Processing**: No dependency on OpenAI for embeddings

### Key Features

- **Knowledge Search**: Agent searches documentation before answering
- **Session Persistence**: Conversations are saved and can be resumed
- **History Context**: Previous conversations are included in context
- **Markdown Output**: Responses are formatted in markdown

## File Structure

```
Level-2-Agents-with-knowledge-and-storage/
├── basic_agent.py                      # Basic Level 2 agent implementation
├── advanced_agent_default_embedder.py  # Advanced agent with LanceDB (default embedder)
├── advanced_agent_huggingface_embeddings.py  # Advanced agent with Huggingface embeddings
├── test_advanced_agent_default_embedder.py   # Test script for advanced agent (default embedder)
├── test_advanced_agent_huggingface_embeddings.py  # Test script for advanced agent (Huggingface embeddings)
├── demo_huggingface_embeddings.py      # Demo script for Huggingface embeddings
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── tmp/                               # Generated files (auto-created)
│   ├── agent.db                      # SQLite session storage (basic)
│   ├── agent_advanced_default.db     # SQLite session storage (advanced default)
│   ├── agent_advanced_huggingface.db # SQLite session storage (advanced huggingface)
│   ├── lancedb_advanced_default/     # LanceDB vector database (advanced default)
│   └── lancedb_advanced_huggingface/ # LanceDB vector database (advanced huggingface)
└── .gitignore                         # Git ignore file
```

## Configuration

### Knowledge Base (Basic)
- **Source**: Agno documentation from `https://docs.agno.com/introduction.md`
- **Vector DB**: Built-in Agno vector storage
- **Embeddings**: Default Agno embeddings (OpenAI compatible)

### Knowledge Base (Advanced)
- **Source**: Agno documentation from `https://docs.agno.com/introduction.md`
- **Vector DB**: LanceDB with hybrid search
- **Embeddings**: Agno default embeddings (OpenAI compatible)

### Knowledge Base (Advanced with Embeddings)
- **Source**: Agno documentation from `https://docs.agno.com/introduction.md`
- **Vector DB**: LanceDB with hybrid search
- **Embeddings**: Custom Huggingface sentence-transformers/all-MiniLM-L6-v2
- **Fallback**: Hash-based embedding system for robustness

### Storage
- **Database**: SQLite with table `agent_sessions`
- **History**: Last 3 conversation runs included in context
- **Persistence**: Automatic session saving

### Agent Settings
- **Model**: DeepSeek Chat
- **Instructions**: Search knowledge before answering
- **Markdown**: Enabled for formatted output
- **Streaming**: Real-time response generation

## Customization

### Change Knowledge Source
```python
knowledge = UrlKnowledge(
    urls=["https://your-docs-url.com"],
    # ... other settings
)
```

### Use Different Model
```python
agent = Agent(
    model=DeepSeek(id="deepseek-coder"),
    # ... other settings
)
```

### Modify Storage
```python
storage = SqliteStorage(
    table_name="custom_sessions", 
    db_file="tmp/custom.db"
)
```

## Troubleshooting

### Common Issues

1. **API Keys**: Ensure both `DEEPSEEK_API_KEY` and `OPENAI_API_KEY` are set
2. **First Run**: Knowledge loading may take time on first execution
3. **Storage**: Check `tmp/` directory for generated files

### Performance Tips

- Knowledge base is loaded once and cached
- Subsequent runs skip knowledge loading
- Use `recreate=True` to rebuild knowledge base if needed

## Next Steps

This Level 2 agent can be extended to:
- Add custom knowledge sources
- Implement Level 3 features (memory and reasoning)
- Create multi-agent systems
- Add custom tools and workflows

## Reference

Based on the [Agno Level 2 documentation](https://docs.agno.com/introduction/agents#level-2%3A-agents-with-knowledge-and-storage). 