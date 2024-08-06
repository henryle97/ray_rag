import json
from pathlib import Path
from typing import Optional

from qdrant_client.conversions import common_types as types
from tqdm import tqdm

from rag import config
from rag.agent import QueryAgent
from rag.index import load_index
from rag.llm import send_request
from rag.schemas import HyperParameter


def generate_responses(
    exp_name: str,
    hyperparams: HyperParameter,
    system_content: str,
    assistant_content: str,
    docs_dir: str,
    exp_dir: str,
    ref_fp: str,
    num_samples: Optional[int] = None,
):
    # Build index
    chunks = load_index(
        embedding_model_name=hyperparams.embedding_model_name,
        embedding_dim=hyperparams.embedding_dim,
        docs_dir=docs_dir,
        chunk_size=hyperparams.chunk_size,
        chunk_overlap=hyperparams.chunk_overlap,
    )

    # lexical index
    lexical_index = None
    if hyperparams.use_lexical_search:
        pass

    # Reranker
    reranker = None

    # query agent
    agent = QueryAgent(
        embedding_model_name=hyperparams.embedding_model_name,
        llm_model_name=hyperparams.llm_model_name,
        temperature=hyperparams.temperature,
        max_context_length=hyperparams.max_context_length,
        system_content=system_content,
        assistant_content=assistant_content,
        lexical_index=lexical_index,
        reranker=reranker,
        chunks=chunks,
    )

    # generate responses
    results = []
    with open(ref_fp, "r") as f:
        questions = [item["question"] for item in json.load(f)][:num_samples]
    for query in tqdm(questions, total=len(questions)):
        response = agent(
            query=query,
            top_k_contexts=hyperparams.num_chunks,
            lexical_search_k=hyperparams.lexical_search_k,
            rerank_threshold=hyperparams.reranking_threshold,
            rerank_k=hyperparams.reranking_k,
            stream=False,
        )
        results.append(response)

    # save to file
    response_fp = Path(
        config.ROOT_DIR, exp_dir, "responses", f"{exp_name}.json"
    )
    response_fp.parent.mkdir(parents=True, exist_ok=True)
    exp_config = {
        "experiment_name": exp_name,
        "chunk_size": hyperparams.chunk_size,
        "chunk_overlap": hyperparams.chunk_overlap,
        "num_chunks": hyperparams.num_chunks,
        "embedding_model_name": hyperparams.embedding_model_name,
        "llm": hyperparams.llm_model_name,
        "temperature": hyperparams.temperature,
        "max_context_length": hyperparams.max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
        "docs_dir": str(docs_dir),
        "experiments_dir": str(exp_dir),
        "references_fp": str(ref_fp),
        "num_samples": len(questions),
    }

    responses = {"config": exp_config, "results": results}
    with open(response_fp, "w") as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)
