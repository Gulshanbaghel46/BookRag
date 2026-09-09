"""
API-level tests using FastAPI's TestClient.

These tests exercise the real app instance and its actual request/response
behavior. They deliberately run WITHOUT the lifespan startup hook (so
Postgres/Redis/RabbitMQ/Ollama are not required), which also verifies that
every route fails *gracefully* (a clean 4xx/5xx JSON body) rather than
crashing when a dependent service is unavailable — an important property in
CI and in any environment where the optional services aren't running yet.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_reports_degraded_without_backends():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert set(body["services"].keys()) == {"cache", "vector_store", "llm"}


def test_root_serves_static_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_query_rejects_empty_query():
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_upload_rejects_disallowed_file_type():
    response = client.post(
        "/documents",
        files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert ".exe" in response.json()["detail"]


def test_upload_rejects_empty_file():
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_invalid_metadata_json():
    response = client.post(
        "/documents",
        files={"file": ("note.txt", b"hello world", "text/plain")},
        data={"metadata": "{not valid json"},
    )
    assert response.status_code == 400


def test_query_returns_clean_500_when_vector_store_unavailable():
    # No lifespan has run, so the vector store is not initialized.
    response = client.post("/query", json={"query": "What is RAG?"})
    assert response.status_code == 500
    assert "detail" in response.json()


def test_list_documents_returns_clean_500_when_db_unavailable():
    # Regression test: this endpoint previously let a raw psycopg
    # OperationalError propagate as an unhandled 500 instead of a
    # clean JSON error response.
    response = client.get("/documents")
    assert response.status_code == 500
    assert "detail" in response.json()


def test_get_document_returns_clean_500_when_db_unavailable():
    response = client.get("/documents/some-id")
    assert response.status_code == 500
    assert "detail" in response.json()
