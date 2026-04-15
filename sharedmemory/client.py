"""SharedMemory client — mirrors the TypeScript SDK API surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import httpx


class SharedMemory:
    """Persistent memory layer for AI agents.

    Usage::

        from sharedmemory import SharedMemory

        memory = SharedMemory(api_key="sm_live_...", volume_id="vol-uuid")
        result = memory.add("The user prefers dark mode")
        results = memory.search("user preferences")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sharedmemory.ai",
        volume_id: str = "default",
        agent_name: str = "python-sdk",
        timeout: float = 30.0,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.volume_id = volume_id
        self.agent_name = agent_name
        self.timeout = timeout
        self.user_id = user_id
        self.agent_id = agent_id
        self.app_id = app_id
        self.session_id = session_id
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SharedMemory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── internal helpers ──

    def _entity_scope(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        return {
            "user_id": user_id or self.user_id,
            "agent_id": agent_id or self.agent_id,
            "app_id": app_id or self.app_id,
            "session_id": session_id or self.session_id,
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
            raise SharedMemoryError(resp.status_code, detail.get("error", str(detail)))
        return resp.json()

    # ── Core Memory Operations ──

    def add(
        self,
        content: str,
        *,
        volume_id: Optional[str] = None,
        memory_type: str = "factual",
        source: str = "sdk",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a memory. Returns processing result with status, confidence, memory_id."""
        return self._request("POST", "/agent/memory/write", json={
            "content": content,
            "volume_id": volume_id or self.volume_id,
            "agent": self.agent_name,
            "memory_type": memory_type,
            "source": source,
            "tags": tags,
            "metadata": metadata,
            **self._entity_scope(user_id, agent_id, app_id, session_id),
        })

    remember = add  # alias

    def search(
        self,
        query: str,
        *,
        volume_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
        rerank_method: Optional[str] = None,
        include_context: bool = False,
        template_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search memories by semantic similarity with optional reranking and context assembly."""
        return self._request("POST", "/agent/memory/query", json={
            "query": query,
            "volume_id": volume_id or self.volume_id,
            "limit": limit,
            "filters": filters,
            "rerank": rerank,
            "rerank_method": rerank_method,
            "include_context": include_context,
            "template_id": template_id,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        })

    recall = search  # alias

    def get(self, memory_id: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a single memory by ID."""
        vol = volume_id or self.volume_id
        return self._request("GET", f"/agent/memory/{memory_id}", params={"volume_id": vol})

    def update(
        self,
        memory_id: str,
        content: str,
        *,
        volume_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a memory's content (creates audit trail)."""
        return self._request("PATCH", f"/agent/memory/{memory_id}", json={
            "volume_id": volume_id or self.volume_id,
            "content": content,
            "metadata": metadata,
        })

    def delete(self, memory_id: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Soft-delete a memory (creates audit trail)."""
        return self._request("DELETE", f"/agent/memory/{memory_id}", json={
            "volume_id": volume_id or self.volume_id,
        })

    def add_many(
        self,
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Batch write up to 100 memories in a single request.

        Each item should have 'content' and optionally 'volume_id', 'memory_type',
        'tags', 'metadata', 'user_id', 'agent_id', 'app_id', 'session_id'.
        """
        payload = []
        for m in memories:
            payload.append({
                "content": m["content"],
                "volume_id": m.get("volume_id", self.volume_id),
                "memory_type": m.get("memory_type", "factual"),
                "tags": m.get("tags"),
                "metadata": m.get("metadata"),
                **self._entity_scope(
                    m.get("user_id"), m.get("agent_id"),
                    m.get("app_id"), m.get("session_id"),
                ),
            })
        return self._request("POST", "/agent/memory/batch", json={"memories": payload})

    # ── Feedback & History ──

    def feedback(
        self,
        memory_id: str,
        feedback: str,
        *,
        volume_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit feedback on a memory ('POSITIVE', 'NEGATIVE', or 'VERY_NEGATIVE')."""
        return self._request("POST", "/memory/feedback", json={
            "memory_id": memory_id,
            "volume_id": volume_id or self.volume_id,
            "feedback": feedback,
            "feedback_reason": reason,
            **self._entity_scope(),
        })

    def history(self, memory_id: str) -> Dict[str, Any]:
        """Get audit trail for a memory."""
        return self._request("GET", f"/memory/feedback/history/{memory_id}")

    # ── Knowledge Graph ──

    def get_entity(self, name: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Get entity details from the knowledge graph."""
        return self._request("POST", "/agent/entity", json={
            "entity_name": name,
            "volume_id": volume_id or self.volume_id,
        })

    def search_entities(
        self,
        query: str,
        *,
        volume_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search entities by name pattern."""
        return self._request("POST", "/agent/entities/search", json={
            "query": query,
            "volume_id": volume_id or self.volume_id,
            "limit": limit,
        })

    def get_graph(self, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the full knowledge graph for a volume."""
        return self._request("POST", "/agent/graph", json={
            "volume_id": volume_id or self.volume_id,
        })

    # ── Context Assembly ──

    def assemble_context(
        self,
        *,
        volume_id: Optional[str] = None,
        template_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble a context block for LLM prompting (Zep-style)."""
        return self._request("POST", "/memory/context/assemble", json={
            "volume_id": volume_id or self.volume_id,
            "template_id": template_id,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        })

    # ── Sessions ──

    def start_session(
        self,
        session_id: str,
        *,
        volume_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new session for scoped memory tracking."""
        return self._request("POST", "/memory/sessions/start", json={
            "session_id": session_id,
            "volume_id": volume_id or self.volume_id,
            "user_id": user_id or self.user_id,
            "agent_id": agent_id or self.agent_id,
            "app_id": app_id or self.app_id,
        })

    def end_session(
        self,
        session_id: str,
        *,
        volume_id: Optional[str] = None,
        auto_summarize: bool = True,
    ) -> Dict[str, Any]:
        """End a session. If auto_summarize=True, compresses session memories into long-term storage."""
        return self._request("POST", "/memory/sessions/end", json={
            "session_id": session_id,
            "volume_id": volume_id or self.volume_id,
            "auto_summarize": auto_summarize,
        })

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details including memory count."""
        return self._request("GET", f"/memory/sessions/{session_id}")

    def list_sessions(
        self,
        *,
        volume_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List sessions for a volume."""
        params: Dict[str, str] = {"volume_id": volume_id or self.volume_id}
        if status:
            params["status"] = status
        return self._request("GET", "/memory/sessions", params=params)

    # ── Export / Import ──

    def export_memories(self, *, volume_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export all memories for a volume."""
        vol = volume_id or self.volume_id
        return self._request("GET", "/agent/memory/export", params={"volume_id": vol})

    def import_memories(
        self,
        memories: List[Dict[str, Any]],
        *,
        volume_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bulk import memories into a volume."""
        return self._request("POST", "/agent/memory/import", json={
            "volume_id": volume_id or self.volume_id,
            "memories": memories,
        })

    # ── Structured Extraction ──

    def extract(
        self,
        text: str,
        schema_id: str,
        *,
        volume_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured data from text using a predefined JSON schema."""
        return self._request("POST", "/memory/extract", json={
            "text": text,
            "volume_id": volume_id or self.volume_id,
            "schema_id": schema_id,
        })


class SharedMemoryError(Exception):
    """Raised when the SharedMemory API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"SharedMemory API error {status_code}: {message}")
