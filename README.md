# SharedMemory Python SDK

The persistent memory layer for AI agents. Add, search, and manage long-term memories with knowledge graph, entity scoping, session lifecycle, and structured extraction.

## Installation

```bash
pip install sharedmemory-ai
```

## Quick Start

```python
from sharedmemory import SharedMemory

memory = SharedMemory(
    api_key="sm_live_...",
    volume_id="your-volume-uuid",
)

# Store a memory
result = memory.remember("User prefers dark mode and compact layout")
print(result["status"])  # "approved"

# Query memories
results = memory.query("what are the user's UI preferences?")
for source in results["sources"]:
    print(source["content"], source["score"])

# Chat (RAG + LLM answer) — the default way to interact
result = memory.chat("What does the user prefer for their UI?")
print(result["answer"])
print(result["sources"])

# Batch write
memory.remember_many([
    {"content": "User's name is Alice"},
    {"content": "Alice works at Acme Corp"},
    {"content": "Alice prefers Python over JavaScript"},
])
```

## Entity Scoping

Scope memories to specific users, agents, or sessions:

```python
memory = SharedMemory(
    api_key="sm_live_...",
    volume_id="vol-uuid",
    user_id="user-123",
    agent_id="chatbot-1",
)

# All operations are automatically scoped
memory.remember("User asked about pricing", session_id="sess-abc")
results = memory.query("pricing", session_id="sess-abc")
```

## Sessions

Track conversation sessions with automatic summarization:

```python
# Start a session
memory.start_session("conv-001", user_id="user-123")

# Add memories during the session
memory.remember("User asked about pricing", session_id="conv-001")
memory.remember("User is interested in enterprise plan", session_id="conv-001")

# End session — automatically summarizes into long-term memory
summary = memory.end_session("conv-001", auto_summarize=True)
```

## Advanced Search

```python
# With reranking
results = memory.query(
    "project deadlines",
    rerank=True,
    rerank_method="llm",
    include_context=True,
)

# With metadata filters
results = memory.query("preferences", filters={
    "AND": [
        {"field": "memory_class", "op": "eq", "value": "preference"},
        {"field": "score", "op": "gte", "value": 0.5},
    ]
})
```

## Knowledge Graph

```python
# Get entity details
entity = memory.get_entity("Alice")
print(entity["summary"])
print(entity["facts"])

# Get full graph
graph = memory.get_graph()
```

## Feedback

```python
memory.feedback("memory-uuid", "POSITIVE", reason="Highly relevant")
memory.feedback("memory-uuid", "NEGATIVE", reason="Outdated information")
```

## Context Assembly

Get an optimized context block for LLM prompting:

```python
context = memory.get_context(template_id="conversational")
print(context["blocks"])  # Ready for system prompt injection
```

## Structured Extraction

```python
data = memory.extract(
    text="Alice is 28, works at Acme Corp as a Senior Engineer since 2022",
    schema_id="contact-info",
)
print(data)  # {"name": "Alice", "age": 28, "company": "Acme Corp", ...}
```

## Async Client

```python
from sharedmemory.async_client import AsyncSharedMemory

async with AsyncSharedMemory(api_key="sm_live_...") as memory:
    await memory.remember("async memory")
    results = await memory.query("query")
```

## API Reference

| Method | Description |
|--------|-------------|
| `remember(content)` | Store a memory |
| `query(query)` | Query memories (semantic search) |
| `chat(query)` | Ask a question — LLM answers using your memories |
| `get(memory_id)` | Get single memory |
| `update(memory_id, content)` | Update memory |
| `delete(memory_id)` | Soft-delete |
| `remember_many(memories)` | Batch write (up to 100) |
| `delete_many(memory_ids)` | Batch delete |
| `update_many(updates)` | Batch update |
| `feedback(memory_id, feedback)` | Quality feedback |
| `history(memory_id)` | Audit trail |
| `get_entity(name)` | Knowledge graph entity |
| `search_entities(query)` | Search entities by name |
| `get_graph()` | Full knowledge graph |
| `list_volumes()` | List accessible volumes |
| `get_context()` | LLM context block |
| `start_session(id)` | Start session |
| `end_session(id)` | End + summarize session |
| `get_session(id)` | Get session details |
| `list_sessions()` | List sessions |
| `webhook_subscribe(url, *, events)` | Register webhook |
| `webhook_unsubscribe(url)` | Remove webhook |
| `export_memories()` | Export all memories |
| `import_memories(memories)` | Bulk import |
| `extract(text, schema_id)` | Structured extraction |
| `create_agent(org_id, project_id, name)` | Create agent |
| `list_agents(org_id)` | List agents |
| `get_agent(agent_id)` | Get agent details |
| `update_agent(agent_id)` | Update agent |
| `delete_agent(agent_id)` | Delete agent |
| `rotate_agent_key(agent_id)` | Rotate agent API key |
| `list_orgs()` | List organizations |
| `get_org(org_id)` | Get org details |
| `list_org_members(org_id)` | List org members |
| `apply_promo(org_id, code)` | Apply promo code |

## License

MIT
