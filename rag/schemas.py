from dataclasses import dataclass


@dataclass
class Record:
    path: str


@dataclass
class Section:
    source: str
    text: str

    def to_dict(self):
        return {"source": self.source, "text": self.text}


@dataclass
class QueryAgentResponse:
    questions: str
    sources: list[str]
    document_ids: list[str]
    answer: str


@dataclass
class QueryAgentWithContentResponse:
    answer: str


@dataclass
class QueryAgentResponse(QueryAgentWithContentResponse):
    questions: str
    sources: list[str]
    document_ids: list[str]


from rag import config


@dataclass
class HyperParameter:
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP
    num_chunks: int = config.TOP_K_CONTEXT
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    embedding_dim: int = config.EMBEDDING_DIMENSIONS[
        config.EMBEDDING_MODEL_NAME
    ]
    llm_model_name: str = config.LLM_MODEL_NAME
    evaluator: str = config.EVALUATOR
    temperature: float = 0.0
    max_context_length: int = 4096
    use_lexical_search: bool = False
    lexical_search_k: int = 1
    use_reranking: bool = False
    reranking_k: int = 7
    reranking_threshold: float = 0.2
