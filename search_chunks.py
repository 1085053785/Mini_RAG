import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba


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


def embed_query(query, model):
    
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return query_embedding[0]


def vector_search(query, embeddings, chunks, model, top_k=3):
    

    query_embedding = embed_query(query, model)

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


def tokenize_text(text):
    tokens = []
    for token in jieba.cut(text):
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens


def build_bm25_index(chunks):
    text = [chunk["text"] for chunk in chunks]
    tokenized_corpus = [tokenize_text(t) for t in text]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search(query, bm25, chunks, top_k=3):
    tokenized_query = tokenize_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_indices:
        if scores[i] <= 0:
            continue
        results.append({"score": float(scores[i]),
        "chunk": chunks[i]})

    return results

def rrf_fusion(vector_results, bm25_results, final_k=3, rrf_k=60):
    fused_results = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk = result["chunk"]
        chunk_key = chunk["text"]
        rrf_score = 1 / (rrf_k + rank)

        fused_results[chunk_key] = {
            "score": rrf_score,
            "chunk": chunk
        }

    for rank, result in enumerate(bm25_results, start=1):
        chunk = result["chunk"]
        chunk_key = chunk["text"]
        rrf_score = 1 / (rrf_k + rank)

        if chunk_key in fused_results:
            fused_results[chunk_key]["score"] += rrf_score
        else:
            fused_results[chunk_key] = {
                "score": rrf_score,
                "chunk": chunk
            }

    results = list(fused_results.values())

    results = sorted(
        results,
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:final_k]

def main():
    model = SentenceTransformer(MODEL_NAME)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    bm25 = build_bm25_index(chunks)

    print("embedding shape:", embeddings.shape)
    print("chunk 数量:", len(chunks))

    query = input("请输入查询内容：")

    vector_results = vector_search(query, embeddings, chunks, model, top_k=5)
    bm25_results = bm25_search(query, bm25, chunks, top_k=5)
    results = rrf_fusion(vector_results, bm25_results, final_k=3, rrf_k=60)

    for i, result in enumerate(results, start=1):
        print("=" * 50)
        print("排名：", i)
        print("RRF 分数：", result["score"])
        print("内容：")
        print(result["chunk"]["text"])

if __name__ == "__main__":
    main()