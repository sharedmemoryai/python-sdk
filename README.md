# SharedMemory Python SDK

The persistent memory layer for AI agents. Add, search, and manage long-term memories with knowledge graph, entity scoping, session lifecycle, and structured extraction.

## Installation

```bash
pip install sharedmemory
```

## Quick Start

```python
from sharedmemory import SharedMemory

memory = SharedMemory(
    api_key="sm_live_...",
    volume_id="your-volume-uuid",
)

# Add a memory
result = memory.add("User prefers dark mode and compact layout")
print(result["status"])  # "approved"

# Search memories
results = memory.search("what are the user's UI preferences?")
for source in results["sources"]:
    print(source["content"], source["score"])

# Batch add
memory.add_many([
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
memory.add("User asked about pricing", session_id="sess-abc")
results = memory.search("pricing", session_id="sess-abc")
```

## Sessions

Track conversation sessions with automatic summarization:

```python
# Start a session
memory.start_session("conv-001", user_id="user-123")

# Add memories during the session
memory.add("User asked about pricing", session_id="conv-001")
memory.add("User is interested in enterprise plan", session_id="conv-001")

# End session — automatically summarizes into long-term memory
summary = memory.end_session("conv-001", auto_summarize=True)
```

## Advanced Search

```python
# With reranking
results = memory.search(
    "project deadlines",
    rerank=True,
    rerank_method="llm",
    include_context=True,
)

# With metadata filters
results = memory.search("preferences", filters={
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
context = memory.assemble_context(template_id="conversational")
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
    await memory.add("async memory")
    results = await memory.search("query")
```

## API Reference

| Method | Description |
|--------|-------------|
| `add(content)` | Add a memory |
| `search(query)` | Semantic search |
| `get(memory_id)` | Get single memory |
| `update(memory_id, content)` | Update memory |
| `delete(memory_id)` | Soft-delete |
| `add_many(memories)` | Batch write (up to 100) |
| `feedback(memory_id, feedback)` | Quality feedback |
| `history(memory_id)` | Audit trail |
| `get_entity(name)` | Knowledge graph entity |
| `get_graph()` | Full knowledge graph |
| `assemble_context()` | LLM context block |
| `start_session(id)` | Start session |
| `end_session(id)` | End + summarize session |
| `export_memories()` | Export all memories |
| `import_memories(data)` | Bulk import |
| `extract(text, schema_id)` | Structured extraction |

## License

MIT
