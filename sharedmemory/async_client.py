"""Async client for SharedMemory.ai — uses httpx.AsyncClient."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .client import SharedMemoryError


class AsyncSharedMemory:
    """Async persistent memory layer for AI agents.

    Usage::

        from sharedmemory.async_client import AsyncSharedMemory

        async with AsyncSharedMemory(api_key="sm_live_...") as memory:
            await memory.add("The user prefers dark mode")
            results = await memory.search("user preferences")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sharedmemory.ai",
        volume_id: str = "",
        agent_name: str = "python-sdk",
        timeout: float = 30.0,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        if not volume_id:
            raise ValueError("volume_id is required. Get yours from the SharedMemory dashboard.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.volume_id = volume_id
        self.agent_name = agent_name
        self.user_id = user_id
        self.agent_id = agent_id
        self.app_id = app_id
        self.session_id = session_id
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncSharedMemory":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _entity_scope(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, str]:
        scope: Dict[str, str] = {}
        v = user_id or self.user_id
        if v: scope["user_id"] = v
        v = agent_id or self.agent_id
        if v: scope["agent_id"] = v
        v = app_id or self.app_id
        if v: scope["app_id"] = v
        v = session_id or self.session_id
        if v: scope["session_id"] = v
        return scope

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = await self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
            raise SharedMemoryError(resp.status_code, detail.get("error", str(detail)))
        return resp.json()

    # ── Core Memory Operations ──

    async def remember(
        self, content: str, *, volume_id: Optional[str] = None,
        memory_type: str = "factual", event_date: Optional[str] = None,
        tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None, agent_id: Optional[str] = None,
        app_id: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory. Returns processing result with status, confidence, memory_id."""
        body: Dict[str, Any] = {
            "content": content,
            "volume_id": volume_id or self.volume_id,
            "memory_type": memory_type,
            **self._entity_scope(user_id, agent_id, app_id, session_id),
        }
        if event_date: body["event_date"] = event_date
        if metadata: body["metadata"] = metadata
        return await self._request("POST", "/agent/memory/write", json=body)

    add = remember  # deprecated alias

    async def query(
        self, query_text: str, *, volume_id: Optional[str] = None, limit: int = 10,
        filters: Optional[Dict[str, Any]] = None, rerank: bool = False,
        rerank_method: Optional[str] = None, include_context: bool = False,
        date_from: Optional[str] = None, date_to: Optional[str] = None,
        template_id: Optional[str] = None, user_id: Optional[str] = None,
        agent_id: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query memories by semantic similarity with optional reranking and context assembly."""
        body: Dict[str, Any] = {
            "query": query_text,
            "volume_id": volume_id or self.volume_id,
            "limit": limit,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        }
        if filters: body["filters"] = filters
        if rerank: body["rerank"] = rerank
        if rerank_method: body["rerank_method"] = rerank_method
        if include_context: body["include_context"] = include_context
        if date_from: body["date_from"] = date_from
        if date_to: body["date_to"] = date_to
        if template_id: body["template_id"] = template_id
        return await self._request("POST", "/agent/memory/query", json=body)

    search = query  # deprecated alias
    recall = query  # deprecated alias

    async def chat(
        self, query: str, *, volume_id: Optional[str] = None, limit: int = 10,
        date_from: Optional[str] = None, date_to: Optional[str] = None,
        rerank: bool = False, user_id: Optional[str] = None,
        agent_id: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask a question and get an LLM-generated answer grounded in your memories.

        Uses the full RAG pipeline: retrieval → reranking → LLM generation → citations.
        Returns {answer, sources, citations}.
        """
        body: Dict[str, Any] = {
            "query": query,
            "volume_id": volume_id or self.volume_id,
            "limit": limit,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        }
        if date_from: body["date_from"] = date_from
        if date_to: body["date_to"] = date_to
        if rerank: body["rerank"] = rerank
        return await self._request("POST", "/agent/memory/chat", json=body)

    async def get(self, memory_id: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("GET", f"/agent/memory/{memory_id}",
                                   params={"volume_id": volume_id or self.volume_id})

    async def update(self, memory_id: str, content: str, *,
                     volume_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._request("PATCH", f"/agent/memory/{memory_id}", json={
            "volume_id": volume_id or self.volume_id, "content": content, "metadata": metadata,
        })

    async def delete(self, memory_id: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("DELETE", f"/agent/memory/{memory_id}",
                                   json={"volume_id": volume_id or self.volume_id})

    async def delete_many(self, memory_ids: List[str], *,
                          volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Batch delete up to 100 memories in a single request."""
        return await self._request("POST", "/agent/memory/delete/batch", json={
            "volume_id": volume_id or self.volume_id,
            "memory_ids": memory_ids,
        })

    async def update_many(self, updates: List[Dict[str, Any]], *,
                          volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Batch update up to 100 memories in a single request.

        Each item should have 'memory_id', 'content', and optionally 'metadata'.
        """
        return await self._request("POST", "/agent/memory/update/batch", json={
            "volume_id": volume_id or self.volume_id,
            "updates": [{"memory_id": u["memory_id"], "content": u["content"],
                         **({"metadata": u["metadata"]} if "metadata" in u else {})}
                        for u in updates],
        })

    async def remember_many(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch write up to 100 memories in a single request.

        Each item should have 'content' and optionally 'volume_id', 'memory_type',
        'tags', 'metadata', 'user_id', 'agent_id', 'app_id', 'session_id'.
        """
        payload = []
        for m in memories:
            item: Dict[str, Any] = {
                "content": m["content"],
                "volume_id": m.get("volume_id", self.volume_id),
                "memory_type": m.get("memory_type", "factual"),
                **self._entity_scope(
                    m.get("user_id"), m.get("agent_id"),
                    m.get("app_id"), m.get("session_id"),
                ),
            }
            if m.get("metadata"): item["metadata"] = m["metadata"]
            payload.append(item)
        return await self._request("POST", "/agent/memory/batch", json={"memories": payload})

    add_many = remember_many  # deprecated alias

    # ── Feedback & History ──

    async def feedback(self, memory_id: str, feedback: str, *,
                       volume_id: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/feedback", json={
            "memory_id": memory_id, "volume_id": volume_id or self.volume_id,
            "feedback": feedback, "feedback_reason": reason, **self._entity_scope(),
        })

    async def history(self, memory_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/agent/memory/feedback/history/{memory_id}")

    # ── Webhooks ──

    async def webhook_subscribe(
        self, url: str, *, volume_id: Optional[str] = None,
        events: Optional[List[str]] = None, secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a persistent HTTP webhook for volume events."""
        body: Dict[str, Any] = {
            "volume_id": volume_id or self.volume_id,
            "url": url,
            "events": events or ["memory.approved", "memory.flagged"],
        }
        if secret: body["secret"] = secret
        return await self._request("POST", "/agent/memory/subscribe", json=body)

    async def webhook_unsubscribe(
        self, url: str, *, volume_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Remove a persistent HTTP webhook subscription."""
        return await self._request("DELETE", "/agent/memory/unsubscribe", json={
            "volume_id": volume_id or self.volume_id, "url": url,
        })

    # ── Knowledge Graph ──

    async def get_entity(self, name: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/entity", json={
            "entity_name": name, "volume_id": volume_id or self.volume_id,
        })

    async def search_entities(
        self, query: str, *, volume_id: Optional[str] = None, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search entities by name pattern."""
        return await self._request("POST", "/agent/entities/search", json={
            "query": query, "volume_id": volume_id or self.volume_id, "limit": limit,
        })

    async def get_graph(self, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/graph", json={
            "volume_id": volume_id or self.volume_id,
        })

    # ── Volumes ──

    async def list_volumes(self) -> List[Dict[str, Any]]:
        """List volumes this API key has access to."""
        return await self._request("GET", "/agent/volumes")

    # ── Context Assembly ──

    async def get_context(self, *, volume_id: Optional[str] = None,
                               query: Optional[str] = None,
                               template_id: Optional[str] = None,
                               user_id: Optional[str] = None,
                               agent_id: Optional[str] = None,
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a context block for LLM prompting."""
        body: Dict[str, Any] = {
            "volume_id": volume_id or self.volume_id,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        }
        if query: body["query"] = query
        if template_id: body["template_id"] = template_id
        return await self._request("POST", "/agent/memory/context/assemble", json=body)

    assemble_context = get_context  # deprecated alias

    # ── Instructions ──

    async def set_instruction(
        self,
        content: str,
        *,
        volume_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an instruction that all agents on this volume will receive in their context."""
        body: Dict[str, Any] = {
            "content": content,
            "volume_id": volume_id or self.volume_id,
            "memory_type": "instruction",
            **self._entity_scope(),
        }
        if metadata: body["metadata"] = metadata
        return await self._request("POST", "/agent/memory/write", json=body)

    async def list_instructions(self, *, volume_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all instructions for a volume."""
        vol = volume_id or self.volume_id
        return await self._request("GET", "/agent/memory/list", params={"volume_id": vol, "memory_type": "instruction"})

    async def delete_instruction(self, memory_id: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete an instruction by memory ID."""
        return await self.delete(memory_id, volume_id=volume_id)

    # ── Profile ──

    async def get_profile(
        self,
        *,
        volume_id: Optional[str] = None,
        user_id: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get a comprehensive profile for a volume or user.

        Returns categorized facts (identity, preferences, expertise, projects),
        relationships, recent activity, instructions, topics, stats,
        and a pre-formatted context_block for LLM injection.
        Cached for 5 minutes. Pass refresh=True to bypass.
        """
        body: Dict[str, Any] = {"volume_id": volume_id or self.volume_id}
        if user_id: body["user_id"] = user_id
        if refresh: body["refresh"] = True
        return await self._request("POST", "/agent/memory/profile", json=body)

    # ── Sessions ──

    async def start_session(self, session_id: str, *, volume_id: Optional[str] = None,
                            user_id: Optional[str] = None, agent_id: Optional[str] = None,
                            app_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/sessions/start", json={
            "session_id": session_id, "volume_id": volume_id or self.volume_id,
            "user_id": user_id or self.user_id, "agent_id": agent_id or self.agent_id,
            "app_id": app_id or self.app_id,
        })

    async def end_session(self, session_id: str, *, volume_id: Optional[str] = None,
                          auto_summarize: bool = True) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/sessions/end", json={
            "session_id": session_id, "volume_id": volume_id or self.volume_id,
            "auto_summarize": auto_summarize,
        })

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/agent/memory/sessions/{session_id}")

    async def list_sessions(
        self, *, volume_id: Optional[str] = None, status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List sessions for a volume."""
        params: Dict[str, str] = {"volume_id": volume_id or self.volume_id}
        if status:
            params["status"] = status
        return await self._request("GET", "/agent/memory/sessions", params=params)

    # ── Export / Import ──

    async def export_memories(self, *, volume_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self._request("GET", "/agent/memory/export",
                                   params={"volume_id": volume_id or self.volume_id})

    async def import_memories(self, memories: List[Dict[str, Any]], *,
                              volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/export/import", json={
            "volume_id": volume_id or self.volume_id, "memories": memories,
        })

    # ── Structured Extraction ──

    async def extract(self, text: str, schema_id: str, *,
                      volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/extract", json={
            "text": text, "volume_id": volume_id or self.volume_id, "schema_id": schema_id,
        })

    async def create_extraction_schema(
        self,
        *,
        schema_id: str,
        name: str,
        json_schema: Dict[str, Any],
        description: Optional[str] = None,
        volume_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new extraction schema."""
        body: Dict[str, Any] = {
            "schema_id": schema_id,
            "name": name,
            "json_schema": json_schema,
            "volume_id": volume_id or self.volume_id,
        }
        if description: body["description"] = description
        return await self._request("POST", "/agent/memory/extract/schemas", json=body)

    async def list_extraction_schemas(self, *, volume_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List extraction schemas for a volume."""
        vol = volume_id or self.volume_id
        return await self._request("GET", f"/agent/memory/extract/schemas", params={"volume_id": vol})

    # ── Agent Profile Management ──

    async def create_agent(
        self, org_id: str, project_id: str, name: str, *,
        description: Optional[str] = None, system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an agent with an auto-generated API key."""
        return await self._request("POST", "/agents", json={
            "org_id": org_id, "project_id": project_id, "name": name,
            "description": description, "system_prompt": system_prompt,
        })

    async def list_agents(self, org_id: str, *, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List agents in an organization."""
        params: Dict[str, str] = {"org_id": org_id}
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/agents", params=params)

    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get a single agent by ID."""
        return await self._request("GET", f"/agents/{agent_id}")

    async def update_agent(
        self, agent_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, system_prompt: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update an agent's name, description, or system prompt."""
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if is_active is not None:
            payload["is_active"] = is_active
        return await self._request("PATCH", f"/agents/{agent_id}", json=payload)

    async def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """Deactivate an agent and revoke its API key."""
        return await self._request("DELETE", f"/agents/{agent_id}")

    async def rotate_agent_key(self, agent_id: str) -> Dict[str, Any]:
        """Rotate an agent's API key."""
        return await self._request("POST", f"/agents/{agent_id}/rotate-key")

    # ── Organization Management ──

    async def list_orgs(self) -> List[Dict[str, Any]]:
        """List organizations the current user belongs to."""
        return await self._request("GET", "/orgs")

    async def get_org(self, org_id: str) -> Dict[str, Any]:
        """Get a single organization by ID."""
        return await self._request("GET", f"/orgs/{org_id}")

    async def list_org_members(self, org_id: str) -> List[Dict[str, Any]]:
        """List members of an organization."""
        return await self._request("GET", f"/orgs/{org_id}/members")

    async def apply_promo(self, org_id: str, code: str) -> Dict[str, Any]:
        """Apply a promo code to an organization."""
        return await self._request("POST", f"/orgs/{org_id}/promo", json={"code": code})
