
from typing import List, Optional
from rag import config
from rag.embed import get_embedding_model
from rag.generate import generate_response
from rag.llm import send_request
from rag.schemas import QueryAgentResponse, QueryAgentWithContentResponse
from rag.search import semantic_search
from rag.utils import get_num_tokens
from rag.vector_db import get_vector_db
from qdrant_client.conversions import common_types as types

class QueryAgent:
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        llm_model_name: str = "gpt-4",
        temperature: float = 0.0,
        max_context_length: int = 4096,
        system_content: str = "",
        assistant_content: str = "",
    ):
        if embedding_model_name is not None:
            self.embdding_model = get_embedding_model(
                embedding_model_name=embedding_model_name,
                model_kwargs={
                    "device": config.EMBEDDING_DEVICE,
                    "trust_remote_code": True,
                },
                encode_kwargs={
                    "device": config.EMBEDDING_DEVICE,
                    "batch_size": config.EMBEDDING_BATCHSIZE,
                },
            )
        self.db_client = get_vector_db(
            host=config.DB_HOST, port=config.DB_PORT, db_type=config.DB_TYPE
        )

        # llm
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.context_length = int(0.5 * max_context_length) - get_num_tokens(
            system_content + assistant_content
        )  # max length of input context :  50% of total context reserved for input
        self.max_tokens = int(
            0.5 * max_context_length
        )  # max num of token of output (the other 50% of total context)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(
        self, query: str, top_k_contexts: int = 5, stream: bool = False
    ) -> QueryAgentResponse:
        # Get top k context
        context_results: List[types.ScoredPoint] = semantic_search(
            embedding_model=self.embdding_model,
            query=query,
            top_k=top_k_contexts,
            db_client=self.db_client,
            collection_name=config.COLLECTION_NAME,
        )

        # Add lexical search results

        # Rerank

        # Generate response
        document_ids = [item.id for item in context_results]
        context = [item.payload["text"] for item in context_results]
        sources = [item.payload["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm_model_name=self.llm_model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=user_content,
            stream=stream,
        )

        return QueryAgentResponse(
            questions=query,
            sources=sources,
            document_ids=document_ids,
            answer=answer,
        )
        
        
class QueryAgentWithContext(QueryAgent):
    def __call__(self, query: str, context: str, stream: bool = False):
        user_content = f"query: {query}, context: {context}"
        response = generate_response(
            llm_model_name=self.llm_model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=user_content[:self.context_length],
            stream=stream,
        )
        return QueryAgentWithContentResponse(
            answer=response
        )
        