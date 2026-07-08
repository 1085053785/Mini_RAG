import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path(r"D:\Dev\PythonWorkspace\documents\embeddings.npy")
CHUNKS_PATH = Path(r"D:\Dev\PythonWorkspace\documents\chunks_embedded.json")

MODEL_NAME = r"D:\Dev\PythonWorkspace\models\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"


def load_embeddings(embeddings_path):
    embeddings = np.load(embeddings_path)
    return embeddings


def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def embed_query(query, model_name):
    model = SentenceTransformer(model_name)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return query_embedding[0]


def search(query, embeddings, chunks, model_name, top_k=3):
    query_embedding = embed_query(query, model_name)

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = {
            "score": float(scores[index]),
            "chunk": chunks[index]
        }
        results.append(result)

    return results


def main():
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    print("embedding shape:", embeddings.shape)
    print("chunk 数量:", len(chunks))

    query = input("请输入查询内容：")

    results = search(query, embeddings, chunks, MODEL_NAME, top_k=3)

    for i, result in enumerate(results, start=1):
        print("=" * 50)
        print("排名：", i)
        print("相似度：", result["score"])
        print("内容：")
        print(result["chunk"]["text"])


if __name__ == "__main__":
    main()