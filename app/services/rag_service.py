import time
import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector

from app.config import get_settings
from app.models import Citation, QueryResponse
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)
settings = get_settings()

GROUNDED_PROMPT = PromptTemplate(
    template=(
        "You are a document Q&A assistant. Answer using ONLY the retrieved context below.\n"
        "If the context does not contain enough information, say that the documents do not "
        "contain that information. Do not use outside knowledge. Do not invent facts, "
        "page numbers, or citations.\n\n"
        "Write the answer in your own words: summarize and explain the relevant facts from the "
        "context instead of copying its sentences verbatim. A short exact phrase is fine only "
        "when the precise wording matters (a defined term, a number, a name); everything else "
        "should be paraphrased. Keep the answer focused and no longer than necessary to fully "
        "answer the question.\n\n"
        "Conversation history may clarify a follow-up question, but it is not evidence. "
        "Never use facts from it unless they are also in the retrieved context.\n\n"
        "Conversation history:\n{history}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["history", "context", "question"],
)


def _distance_to_relevance(distance: float) -> float:
    """Map pgvector distance to a 0-1 relevance score (higher is better)."""
    try:
        return round(1.0 / (1.0 + max(float(distance), 0.0)), 4)
    except (TypeError, ValueError):
        return 0.0


class RAGService:
    def __init__(self):
        if settings.LLM_PROVIDER == "ollama":
            self.llm = ChatOllama(
                model=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE,
                num_predict=settings.MAX_TOKENS,
            )
            logger.info("Using Ollama model: %s", settings.MODEL_NAME)
        else:
            self.llm = ChatOpenAI(
                model=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            logger.info("Using OpenAI model: %s", settings.MODEL_NAME)

        self.vector_store: Optional[PGVector] = None
        self.prompt_template = GROUNDED_PROMPT
        self.memory_store: Dict[str, List[Dict[str, str]]] = {}

    async def initialize(self, vector_store: PGVector):
        self.vector_store = vector_store
        logger.info("RAG service initialized")

    def _history_for(self, session_id: str) -> List[Dict[str, str]]:
        return self.memory_store.setdefault(session_id, [])

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "No prior conversation."
        return "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in history[-3:]
        )

    async def query(
        self,
        query: str,
        session_id: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> QueryResponse:
        start_time = time.time()
        history = self._history_for(session_id)
        history_text = self._format_history(history)

        search_filters = dict(filters or {})
        if document_id:
            search_filters["document_id"] = document_id

        cache_key = cache_service._generate_cache_key(
            "query",
            {
                "query": query,
                "history": history_text,
                "filters": search_filters,
                "top_k": top_k,
            },
        )
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            cached_result["processing_time_ms"] = (time.time() - start_time) * 1000
            cached_result["session_id"] = session_id
            return QueryResponse(**cached_result)

        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized")

        search_kwargs: Dict[str, Any] = {"k": top_k}
        if search_filters:
            search_kwargs["filter"] = search_filters
        docs_with_scores = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            **search_kwargs,
        )
        logger.info("Similarity search returned %s chunks", len(docs_with_scores))

        max_distance = max(0.0, 1.0 - float(settings.SIMILARITY_THRESHOLD))
        kept: List[tuple] = [
            (doc, float(score))
            for doc, score in docs_with_scores
            if float(score) <= max_distance
        ]
        if not kept and docs_with_scores:
            best_doc, best_score = docs_with_scores[0]
            if float(best_score) <= max_distance + 0.35:
                kept = [(best_doc, float(best_score))]

        if not kept:
            response = QueryResponse(
                answer="I could not find relevant information in the uploaded documents to answer that question.",
                citations=[],
                session_id=session_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=settings.MODEL_NAME,
                grounded=True,
            )
            history.append({"question": query, "answer": response.answer})
            del history[:-3]
            return response

        context = "\n\n".join(
            f"[Source {index + 1} | {doc.metadata.get('file_name', 'document')} | "
            f"page {doc.metadata.get('page', 'n/a')}]\n{doc.page_content}"
            for index, (doc, _) in enumerate(kept)
        )
        prompt = self.prompt_template.format(
            history=history_text,
            context=context,
            question=query,
        )
        llm_response = await self.llm.ainvoke(prompt)
        answer_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        citations: List[Citation] = []
        for doc, distance in kept:
            page = doc.metadata.get("page")
            try:
                page_number = int(page) if page is not None else None
            except (TypeError, ValueError):
                page_number = None
            text = doc.page_content or ""
            citations.append(
                Citation(
                    document_id=str(doc.metadata.get("document_id", "unknown")),
                    document_name=str(doc.metadata.get("file_name", "unknown")),
                    page_number=page_number,
                    chunk_text=text[:400] + ("..." if len(text) > 400 else ""),
                    relevance_score=_distance_to_relevance(distance),
                )
            )

        response = QueryResponse(
            answer=answer_text,
            citations=citations,
            session_id=session_id,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used=settings.MODEL_NAME,
            grounded=True,
        )
        await cache_service.set(cache_key, response.model_dump(mode="json"), ttl=settings.CACHE_TTL)
        history.append({"question": query, "answer": answer_text})
        del history[:-3]
        logger.info("Query processed in %.2fms", response.processing_time_ms)
        return response

    async def clear_session(self, session_id: str):
        self.memory_store.pop(session_id, None)
        logger.info("Cleared session: %s", session_id)


rag_service = RAGService()