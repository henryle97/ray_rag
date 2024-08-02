import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct


def get_vector_db(host: str, port: int, db_type: str = "qdrant"):
    return VectorDB(host=host, port=port, db_type=db_type)


class VectorDB:
    def __init__(self, host: str, port: int, db_type: str = "qdrant"):
        self.client = QdrantClient(host=host, port=port, timeout=600)

    def setup(
        self,
        collection_name: str = "default_collection",
        vector_size: int = 768,
        force_recreate: bool = False,
    ):

        vector_config = {
            "embedding": rest.VectorParams(
                distance=rest.Distance.COSINE, size=vector_size
            )
        }
        if not self._is_exists_collection(collection_name):
            print(
                f"Not found collection {collection_name} in {self.client.get_collections()}"
            )
            print(f"Creating collection {collection_name}")
            self.client.create_collection(
                collection_name=collection_name, vectors_config=vector_config
            )
        else:
            if force_recreate:
                print(f"Recreating collection {collection_name}")
                self.client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=vector_config,
                )
            else:
                print(
                    f"Collection {collection_name} already exists, skipping setup"
                )

    def _is_exists_collection(self, collection_name: str):
        client_collections: types.CollectionsResponse = (
            self.client.get_collections()
        )
        client_collections_names = [
            collection.name for collection in client_collections.collections
        ]
        print(f"Client collections: {client_collections_names}")
        is_exists = collection_name in client_collections_names
        return is_exists

    def insert(self, batch: dict, collection_name: str = "default_collection"):

        for text, source, embedding in zip(
            batch["text"], batch["source"], batch["embeddings"]
        ):
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=uuid.uuid4().hex,
                        vector={"embedding": embedding},
                        payload={"source": source, "text": text},
                    )
                ],
            )

    def search(
        self,
        vector: List[float],
        vector_name: str = "embedding",
        collection_name: str = "default_collection",
        top_k: int = 10,
    ) -> List[types.ScoredPoint]:
        query_results = self.client.search(
            collection_name=collection_name,
            query_vector=(vector_name, vector),
            limit=top_k,
            query_filter=None,
        )
        return query_results


if __name__ == "__main__":
    client = VectorDB(host="localhost", port=6345)
    print(client._is_exists_collection(collection_name="test"))
