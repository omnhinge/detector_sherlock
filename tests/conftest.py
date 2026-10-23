import asyncio

import httpx
import pytest

from app import nli_engine


@pytest.fixture(autouse=True)
def disable_external_services(monkeypatch):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def reject_http_requests(self, url, *args, **kwargs):
        raise httpx.ConnectError("External HTTP access is disabled in tests")

    monkeypatch.setattr(httpx.AsyncClient, "get", reject_http_requests)
    monkeypatch.setattr(nli_engine, "_load_model", lambda: None)
    monkeypatch.setattr(nli_engine, "_model", None)
    monkeypatch.setattr(nli_engine, "_model_loaded", True)
    monkeypatch.setattr(nli_engine, "_model_available", False)
    yield

    loop.close()
    asyncio.set_event_loop(None)
