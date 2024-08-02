import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

# Directories
EFS_DIR = Path("/mnt/hdd4T/henryle/llm/dataset")
ROOT_DIR = Path(__file__).parent.parent.absolute()

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cuda:0"
EMBEDDING_BATCHSIZE = 100

# Index
PARQUET_FILE = "/home/hoanglv/works/LLM/RAG/ray_rag/experiments/chunks.parquet"

# Qdrant
COLLECTION_NAME = "ray_docs"
DB_TYPE = "qdrant"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 6333)

# LLM
LLM_API_URL = "http://107.120.94.21:9113/v1"
LLM_MODEL_NAME = (
    "/home/tuyendt/workspaces/huggingface/Meta-Llama-3-8B-Instruct/"
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
TEMPERATURE = 0.0


# RAG CONFIG
TOP_K_CONTEXT = 5
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    "BAAI/bge-large-en": 1024,
    "text-embedding-ada-002": 1536,
    "gte-large-fine-tuned": 1024,
    "BAAI/bge-small-en-v1.5": 384,
}

# Maximum context lengths
MAX_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4-1106-preview": 128000,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "meta-llama/Llama-3-8b-chat-hf": 8192,
    "meta-llama/Llama-3-70b-chat-hf": 8192,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "mistralai/Mistral-7B-Instruct-v0.1": 65536,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 65536,
    "/home/tuyendt/workspaces/huggingface/Meta-Llama-3-8B-Instruct/": 8192,
}

# Pricing per 1M tokens
PRICING = {
    "gpt-3.5-turbo": {"prompt": 1.5, "sampled": 2},
    "gpt-4": {"prompt": 30, "sampled": 60},
    "gpt-4-1106-preview": {"prompt": 10, "sampled": 30},
    "llama-2-7b-chat-hf": {"prompt": 0.15, "sampled": 0.15},
    "llama-2-13b-chat-hf": {"prompt": 0.25, "sampled": 0.25},
    "llama-2-70b-chat-hf": {"prompt": 1, "sampled": 1},
    "llama-3-8b-chat-hf": {"prompt": 0.15, "sampled": 0.15},
    "llama-3-70b-chat-hf": {"prompt": 1, "sampled": 1},
    "codellama-34b-instruct-hf": {"prompt": 1, "sampled": 1},
    "mistral-7b-instruct-v0.1": {"prompt": 0.15, "sampled": 0.15},
    "mixtral-8x7b-instruct-v0.1": {"prompt": 0.50, "sampled": 0.50},
    "mixtral-8x22b-instruct-v0.1": {"prompt": 0.9, "sampled": 0.9},
}


# LLM Inputs
SYSTEM_CONTENT = (
    "Answer the query using the context provided. Be succinct. "
    "Contexts are organized in a list of dictionaries [{'text': <context>}, {'text': <context>}, ...]. "
    "Feel free to ignore any contexts in the list that don't seem relevant to the query. "
)

ASSISTANT_CONTENT = ""

# Evaluation
EXPERIMENTS_DIR = "./experiments"
REFERENCES_FILE_PATH = EXPERIMENTS_DIR + "/references/" + "llama-3.json"
NUM_EVAL_SAMPLES = None   # None = all samples

EVALUATE_SYSTEM_CONTENT = """
    Answer the query using the context provided. Be succinct.
    Then, you must {score} your response between 1 and 5.
    You must return your response in a line with only the score.
    Do not add any more details.
    On a separate line provide your {reasoning} for the score as well.
    Return your response following the exact format outlined below.
    Do not add or remove anything.
    And all of this must be in a valid JSON format.
    
    {"answer": answer,
     "score": score,
     "reasoning": reasoning}
    """

EVALUATE_ASSISTANT_CONTENT = ""