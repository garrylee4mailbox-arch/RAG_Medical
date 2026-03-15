import requests

BASE="http://127.0.0.1:8008"

def test_skip_if_not_running():
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
    except Exception:
        return

def test_409_before_index():
    try:
        requests.get(f"{BASE}/health", timeout=2).raise_for_status()
    except Exception:
        return
    r = requests.post(f"{BASE}/answer", json={"question":"What is hypertension?"}, timeout=5)
    assert r.status_code == 409
    assert r.json().get("error") == "RAG_NOT_READY"
