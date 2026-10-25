from app.main import home, health, verify, VerifyRequest


def _run_coroutine_sync(coro):
    """Run an async coroutine that does not block on I/O synchronously."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not complete synchronously")


def test_home_endpoint():
    data = home()
    assert data["project"] == "Detector Sherlock"
    assert "running" in data["message"]


def test_health_endpoint():
    data = health()
    assert data["status"] == "healthy"
    assert data["version"] == "0.4.0"
    assert data["features"]["wikipedia"] is True
    assert data["features"]["factcheck"] is True
    assert data["features"]["realtime_news"] is True


def test_verify_empty():
    request = VerifyRequest(response="Short.")
    data = _run_coroutine_sync(verify(request))
    assert data["status"] == "completed"
    assert data["claims_count"] == 0
    assert data["summary"]["reliability_score"] == 0.0
    assert data["summary"]["inconclusive"] == 0
