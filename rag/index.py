import os
from functools import partial
from pathlib import Path
from typing import List

import ray
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import rag.config as config
from rag.data import extract_sections
from rag.embed import EmbedChunks
from rag.vector_db import VectorDB, get_vector_db

load_dotenv()


class StoreResults:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        db_name: str = "default",
    ):
        self.db_client = get_vector_db(host=host, port=port, db_name=db_name)
        self.collection_name = collection_name

    def __call__(self, batch: dict):
        self.db_client.insert(batch, collection_name=self.collection_name)
        return {}


def _get_text_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str] = ["\n\n", "\n", " ", ""],
):
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


def chunk_section(
    section: dict, text_splitter: RecursiveCharacterTextSplitter
):
    chunks: List[Document] = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [
        {"text": chunk.page_content, "source": chunk.metadata["source"]}
        for chunk in chunks
    ]


def build_index(
    docs_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    force_recreate: bool = True,
):
    # docs -> sections -> chunks
    print(f"Build index for docs in {docs_dir}")
    # if isinstance(docs_dir, str):
    #     docs_dir = Path(docs_dir)

    # html_paths = [path for path in docs_dir.rglob("*.html") if not path.is_dir()]
    # print(f"Total html files: {len(html_paths)}")

    # ds_items = [{"path": path} for path in html_paths]
    # ds: ray.data.Dataset = ray.data.from_items(ds_items)

    # sections_ds: ray.data.Dataset = ds.flat_map(fn=extract_sections)   # flat_map - Apply the given function to each row and then flatten results.

    # text_splitter = _get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # chunks_ds: ray.data.Dataset = sections_ds.flat_map(
    #     fn=partial(chunk_section, text_splitter=text_splitter)
    # )
    # print(f"Total sections: {sections_ds.count()}")
    # print(f"Total chunks: {chunks_ds.count()}")

    # chunks_ds.write_parquet("/home/hoanglv/works/LLM/RAG/ray_rag/experiments/chunks.parquet")

    chunks_ds = ray.data.read_parquet(
        "/home/hoanglv/works/LLM/RAG/ray_rag/experiments/chunks.parquet"
    )
    print(f"Total chunks: {chunks_ds.count()}")
    # # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={
            "model_name": embedding_model_name,
            "device": config.EMBEDDING_DEVICE,
        },
        batch_size=100,
        num_gpus=1,
        concurrency=1,
    )

    # # Index data
    db_client = get_vector_db(host=config.DB_HOST, port=config.DB_PORT)
    db_client.setup(
        collection_name=config.COLLECTION_NAME,
        vector_size=config.EMBEDDING_DIMENSIONS[config.EMBEDDING_MODEL_NAME],
        force_recreate=force_recreate,
    )

    embedded_chunks.map_batches(
        StoreResults,
        fn_constructor_kwargs={
            "host": config.DB_HOST,
            "port": config.DB_PORT,
            "collection_name": config.COLLECTION_NAME,
        },
        batch_size=128,
        num_cpus=4,
        concurrency=4,
    ).count()

    print("Updated the index!")


if __name__ == "__main__":

    DOCS_DIR = Path(config.EFS_DIR, "docs.ray.io/en/master/")

    build_index(
        docs_dir=DOCS_DIR,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
    )
