from typing import List

from qdrant_client.conversions import common_types as types

from rag.embed import HuggingFaceEmbeddings
from rag.vector_db import VectorDB


def semantic_search(
    query: str,
    db_client: VectorDB,
    embedding_model: HuggingFaceEmbeddings,
    top_k: int,
    collection_name: str,
) -> List[types.ScoredPoint]:
    # Encode the query
    query_embedding = embedding_model.embed_query(text=query)

    # Perform the search
    search_results = db_client.search(
        vector=query_embedding, top_k=top_k, collection_name=collection_name
    )
    return search_results
