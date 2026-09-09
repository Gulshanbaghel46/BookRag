from app.models import QueryRequest, DocumentStatus


def test_query_strips_whitespace():
    req = QueryRequest(query="  What is RAG?  ")
    assert req.query == "What is RAG?"


def test_document_status_values():
    assert DocumentStatus.PENDING.value == "pending"
    assert DocumentStatus.COMPLETED.value == "completed"
