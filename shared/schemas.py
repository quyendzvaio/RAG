from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str
    status: str = "ok"


class ContextItem(BaseModel):
    id: str
    source: str
    title: str = ""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    citation: str = ""


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    candidate_k: int = 40
    use_llm: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    candidate_k: int = 40


class ServiceAnswer(BaseModel):
    service: str
    question: str
    answer: str
    contexts: List[ContextItem] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1200


class ChatResponse(BaseModel):
    model: str
    content: str
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    fallback: bool = False
    error: Optional[str] = None


class MetricScore(BaseModel):
    name: str
    value: float
    explanation: str = ""


class EvaluationRecord(BaseModel):
    question: str
    vector_answer: ServiceAnswer
    wiki_answer: ServiceAnswer
    metrics: List[MetricScore] = Field(default_factory=list)
    preferred: Optional[str] = None


class CompareResponse(BaseModel):
    question: str
    vector: ServiceAnswer
    wiki: ServiceAnswer
    metrics: List[MetricScore] = Field(default_factory=list)
    run_id: Optional[int] = None
