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
    ) -> Dict[str, Optional[str]]:
        return {
            "user_id": user_id or self.user_id,
            "agent_id": agent_id or self.agent_id,
            "app_id": app_id or self.app_id,
            "session_id": session_id or self.session_id,
        }

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = await self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
            raise SharedMemoryError(resp.status_code, detail.get("error", str(detail)))
        return resp.json()

    # ── Core Memory Operations ──

    async def add(
        self, content: str, *, volume_id: Optional[str] = None,
        memory_type: str = "factual", source: str = "sdk",
        tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None, agent_id: Optional[str] = None,
        app_id: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/write", json={
            "content": content, "volume_id": volume_id or self.volume_id,
            "agent": self.agent_name, "memory_type": memory_type, "source": source,
            "tags": tags, "metadata": metadata,
            **self._entity_scope(user_id, agent_id, app_id, session_id),
        })

    remember = add

    async def search(
        self, query: str, *, volume_id: Optional[str] = None, limit: int = 10,
        filters: Optional[Dict[str, Any]] = None, rerank: bool = False,
        rerank_method: Optional[str] = None, include_context: bool = False,
        template_id: Optional[str] = None, user_id: Optional[str] = None,
        agent_id: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/query", json={
            "query": query, "volume_id": volume_id or self.volume_id,
            "limit": limit, "filters": filters, "rerank": rerank,
            "rerank_method": rerank_method, "include_context": include_context,
            "template_id": template_id,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        })

    recall = search

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

    async def add_many(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = [{
            "content": m["content"], "volume_id": m.get("volume_id", self.volume_id),
            "memory_type": m.get("memory_type", "factual"), "tags": m.get("tags"),
            "metadata": m.get("metadata"),
            **self._entity_scope(m.get("user_id"), m.get("agent_id"), m.get("app_id"), m.get("session_id")),
        } for m in memories]
        return await self._request("POST", "/agent/memory/batch", json={"memories": payload})

    # ── Feedback & History ──

    async def feedback(self, memory_id: str, feedback: str, *,
                       volume_id: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/memory/feedback", json={
            "memory_id": memory_id, "volume_id": volume_id or self.volume_id,
            "feedback": feedback, "feedback_reason": reason, **self._entity_scope(),
        })

    async def history(self, memory_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/memory/feedback/history/{memory_id}")

    # ── Knowledge Graph ──

    async def get_entity(self, name: str, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/entity", json={
            "entity_name": name, "volume_id": volume_id or self.volume_id,
        })

    async def get_graph(self, *, volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/graph", json={
            "volume_id": volume_id or self.volume_id,
        })

    # ── Context Assembly ──

    async def assemble_context(self, *, volume_id: Optional[str] = None,
                               template_id: Optional[str] = None,
                               user_id: Optional[str] = None,
                               agent_id: Optional[str] = None,
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/memory/context/assemble", json={
            "volume_id": volume_id or self.volume_id, "template_id": template_id,
            **self._entity_scope(user_id, agent_id, session_id=session_id),
        })

    # ── Sessions ──

    async def start_session(self, session_id: str, *, volume_id: Optional[str] = None,
                            user_id: Optional[str] = None, agent_id: Optional[str] = None,
                            app_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/memory/sessions/start", json={
            "session_id": session_id, "volume_id": volume_id or self.volume_id,
            "user_id": user_id or self.user_id, "agent_id": agent_id or self.agent_id,
            "app_id": app_id or self.app_id,
        })

    async def end_session(self, session_id: str, *, volume_id: Optional[str] = None,
                          auto_summarize: bool = True) -> Dict[str, Any]:
        return await self._request("POST", "/memory/sessions/end", json={
            "session_id": session_id, "volume_id": volume_id or self.volume_id,
            "auto_summarize": auto_summarize,
        })

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/memory/sessions/{session_id}")

    # ── Export / Import ──

    async def export_memories(self, *, volume_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self._request("GET", "/agent/memory/export",
                                   params={"volume_id": volume_id or self.volume_id})

    async def import_memories(self, memories: List[Dict[str, Any]], *,
                              volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/agent/memory/import", json={
            "volume_id": volume_id or self.volume_id, "memories": memories,
        })

    # ── Structured Extraction ──

    async def extract(self, text: str, schema_id: str, *,
                      volume_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/memory/extract", json={
            "text": text, "volume_id": volume_id or self.volume_id, "schema_id": schema_id,
        })
