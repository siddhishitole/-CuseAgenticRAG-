# CuseAgenticRAG 🤖

An intelligent Agentic RAG (Retrieval-Augmented Generation) system with multiple specialized LangGraph agents, automatic intelligent routing, shared memory, and external workflow automation capabilities. Built with LangChain, LangGraph, and powered by OpenAI/Google Gemini models.

## 🌟 Features

### Core Components
- **🧠 Intelligent Agent Router**: LLM-powered routing system that automatically selects the best agent for your query
- **💾 Shared Memory Store**: Cross-agent memory using Memori for context retention
- **🗃️ Vector Database**: Local Chroma vector store with OpenAI embeddings for semantic search
- **🔄 Multi-Model Support**: Flexible LLM configuration supporting OpenAI GPT and Google Gemini models
- **🌐 Web Search Integration**: Tavily and Perplexity API support for real-time information retrieval
- **⚡ CLI Interface**: Interactive and single-shot modes with rich console output

### Specialized Agents
1. **Corrective RAG Agent** 🎯
   - Ideal for straightforward, single-question queries
   - Grades retrieved documents for relevance
   - Automatically falls back to web search when local knowledge is insufficient

2. **Pre-Act RAG Agent** 🧩 ✨
   - Handles complex, multi-faceted queries requiring sequential reasoning
   - Creates comprehensive multi-step plans before execution
   - Dynamically revises plans based on execution outcomes
   - Accumulates context across all planning and execution steps

3. **Workflow Agent** 🔧
   - Integrates with external services for task execution
   - Handles: email automation, calendar scheduling, task creation, reminders
   - Perfect for: "Schedule a meeting for next Monday" or "Send an email to the team"

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- OpenAI API key and/or Google API key
- Tavily API key for web search
- Perplexity API key for advanced search

### Installation

**1. Clone the repository**
```powershell
git clone https://github.com/vinaytiparadi/CuseAgenticRag.git
cd CuseAgenticRag
```

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```powershell
pip install -r requirements.txt
```

**4. Configure environment variables**

```powershell
copy .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY (required)
# - GOOGLE_API_KEY (required for Gemini routing)
# - TAVILY_API_KEY (required)
# - PERPLEXITY_API_KEY (required)
```

**5. Ingest documents (optional)**

Add your documents to the `data/` folder and run:

```powershell
python -m agenticrag.scripts.ingest --path .\data
```

This will create embeddings and store them in the Chroma vector database.

## 💡 Usage

### Basic Query (Automatic Agent Selection)
The system automatically routes your query to the most appropriate agent:

```powershell
python -m agenticrag.cli "What is machine learning?"
```

### Explicitly Select an Agent

**Use Corrective RAG for simple queries:**
```powershell
python -m agenticrag.cli "What is machine learning?" --agent corrective
```

**Use Pre-Act RAG for complex, multi-step queries:**
```powershell
python -m agenticrag.cli "First explain quantum computing basics, then describe its applications in cryptography" --agent preact
```

**Use Workflow Agent for task automation:**
```powershell
python -m agenticrag.cli "Schedule a meeting with the team for next Monday at 3 PM" --agent workflow
```

### Interactive Mode

Start an interactive chat session with automatic agent routing:

```powershell
python -m agenticrag.cli --interactive
```

In interactive mode:
- Type your questions naturally
- The system automatically selects the best agent
- Type `exit` or `quit` to end the session
- Chat history is maintained across the session

### Example Queries

**Simple queries (Corrective Agent):**
- "Who is the CEO of OpenAI?"
- "What are the benefits of exercise?"
- "Explain the concept of neural networks"

**Complex queries (Pre-Act Agent):**
- "Research the history of AI, explain current applications of AI, and predict future impacts of AI"
- "How do I build a RAG system? Explain architecture, implementation, and deployment of RAG"

**Task automation queries (Workflow Agent):**
- "Send an email to john@example.com about the project deadline"
- "Create a calendar event for the client presentation tomorrow"
- "Schedule a reminder for the standup meeting"

## 📁 Project Structure

```
CuseAgenticRag/
├── agenticrag/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                 # CLI interface and interactive mode
│   ├── config.py              # Configuration and settings management
│   ├── llm.py                 # LLM provider abstraction
│   ├── router.py              # Intelligent agent routing system
│   ├── vectorstore.py         # Chroma vector store integration
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── corrective.py      # Corrective RAG agent implementation
│   │   ├── preact.py          # Pre-Act RAG agent implementation
│   │   ├── workflow.py        # Workflow automation agent
│   │   ├── PREACT_README.md   # Detailed Pre-Act documentation
│   │   └── PREACT_QUICKREF.md # Pre-Act quick reference
│   ├── scripts/
│   │   └── ingest.py          # Document ingestion script
│   └── tools/
│       ├── __init__.py
│       ├── n8n_workflow.py    # n8n workflow integration
│       └── websearch.py       # Web search tool integrations
├── chromadb/                  # Local vector database
├── data/                      # Document storage for ingestion
├── .env                       # Environment variables (create from .env.example)
├── .env.example               # Example environment configuration
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
└── README.md                  # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from `.env.example` and configure the following:

**Required:**
```bash
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

**Optional:**
```bash
# Model Configuration
OPENAI_MODEL=gpt-5-nano-2025-08-07           # Default OpenAI model
EMBEDDING_MODEL=text-embedding-3-small        # Embedding model for vector store

# Storage Configuration
CHROMA_PATH=./chromadb                        # Path to Chroma database
MEMORY_NAMESPACE=default                      # Memory namespace for shared context
SESSION_ID=local-dev                          # Session identifier

# Web Search (optional)
TAVILY_API_KEY=your_tavily_key               # For Tavily web search
PERPLEXITY_API_KEY=your_perplexity_key       # For Perplexity API
PERPLEXITY_BASE_URL=https://api.perplexity.ai
```

## 🔍 How It Works

### Intelligent Routing System

The router uses a Gemini-powered LLM to analyze incoming queries and automatically select the best agent based on:

1. **Query Complexity**: Number of questions and reasoning steps required
2. **Query Intent**: Information retrieval vs. task execution vs. complex analysis
3. **Query Scope**: Single-focus vs. multi-faceted queries

**Routing Decision Flow:**
```
User Query → Router Analysis → Agent Selection → Task Execution → Response
```

The router considers:
- Query length and structure
- Presence of multiple questions
- Keywords indicating automation needs
- Complexity indicators (step-by-step, planning, etc.)

### Agent Capabilities

| Agent | Best For | Key Features |
|-------|----------|--------------|
| **Corrective** | Simple Q&A | Document grading, web search fallback |
| **Pre-Act** | Complex reasoning | Multi-step planning, plan revision, context accumulation |
| **Workflow** | Task automation | n8n integration, external service actions |

## 🛠️ Development

### Adding Custom Agents

1. Create a new agent file in `agenticrag/agents/`
2. Implement using LangGraph StateGraph
3. Register in `router.py` by updating:
   - `AgentName` literal type
   - `get_graph()` function
   - Router system prompt with agent description

### Adding Custom Tools

1. Create tool implementation in `agenticrag/tools/`
2. Import and integrate in relevant agent files
3. Update agent prompts to include tool usage instructions

### Running Tests

```powershell
# Test specific agent (if test files exist)
python test_preact.py all
```