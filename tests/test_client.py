"""Basic tests for SharedMemory Python SDK."""

from unittest.mock import MagicMock, patch

import pytest

from sharedmemory import SharedMemory
from sharedmemory.client import SharedMemoryError


def test_init_requires_api_key():
    with pytest.raises(ValueError, match="api_key is required"):
        SharedMemory(api_key="")


def test_init_defaults():
    m = SharedMemory(api_key="sm_live_test")
    assert m.base_url == "https://api.sharedmemory.ai"
    assert m.volume_id == "default"
    assert m.agent_name == "python-sdk"
    m.close()


def test_entity_scope_defaults():
    m = SharedMemory(api_key="test", user_id="u1", agent_id="a1")
    scope = m._entity_scope()
    assert scope["user_id"] == "u1"
    assert scope["agent_id"] == "a1"
    assert scope["app_id"] is None
    m.close()


def test_entity_scope_override():
    m = SharedMemory(api_key="test", user_id="u1")
    scope = m._entity_scope(user_id="u2")
    assert scope["user_id"] == "u2"
    m.close()


def test_context_manager():
    with SharedMemory(api_key="test") as m:
        assert m.api_key == "test"


def test_error_class():
    err = SharedMemoryError(400, "bad request")
    assert err.status_code == 400
    assert "400" in str(err)
    assert "bad request" in str(err)
