from pathlib import Path


EMBEDDINGS_PATH = Path(
    r"D:\Dev\PythonWorkspace\documents\embeddings.npy"
)

CHUNKS_PATH = Path(
    r"D:\Dev\PythonWorkspace\documents\chunks_embedded.json"
)

EMBEDDING_MODEL_PATH = (
    r"D:\Dev\PythonWorkspace\models"
    r"\models--BAAI--bge-small-zh-v1.5"
    r"\snapshots"
    r"\7999e1d3359715c523056ef9478215996d62a620"
)

LLM_MODEL_PATH = (
    r"D:\Dev\PythonWorkspace\models"
    r"\Qwen2.5-0.5B-Instruct"
)

RERANKER_MODEL_PATH = (
    r"D:\Dev\PythonWorkspace\models"
    r"\bge-reranker-v2-m3"
)

MIN_SCORE = 0.45

VECTOR_TOP_K = 5
BM25_TOP_K = 5
FUSION_TOP_K = 5
FINAL_TOP_K = 3

MAX_HISTORY_TURNS = 3