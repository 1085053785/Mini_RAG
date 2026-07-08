import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path(r"D:\Dev\PythonWorkspace\documents\chunks.json")
EMBEDDINGS_PATH = Path(r"D:\Dev\PythonWorkspace\documents\embeddings.npy")
CHUNKS_OUTPUT_PATH = Path(r"D:\Dev\PythonWorkspace\documents\chunks_embedded.json")
MANIFEST_PATH = Path(r"D:\Dev\PythonWorkspace\documents\embedding_manifest.json")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def prepare_texts(chunks):
    valid_chunks = []
    texts = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if text == "":
            continue

        valid_chunks.append(chunk)
        texts.append(text)

    return valid_chunks, texts


def embed_texts(texts, model_name):
    model = SentenceTransformer(
        model_name,
        cache_folder=r"D:\Dev\PythonWorkspace\models"
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings


def save_outputs(valid_chunks, embeddings, model_name):
    np.save(EMBEDDINGS_PATH, embeddings)

    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(valid_chunks, f, ensure_ascii=False, indent=2)

    manifest = {
        "embedding_model": model_name,
        "chunk_count": len(valid_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_path": str(EMBEDDINGS_PATH),
        "chunks_path": str(CHUNKS_OUTPUT_PATH)
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    chunks = load_chunks(CHUNKS_PATH)

    print("读取到的 chunk 数量：", len(chunks))

    valid_chunks, texts = prepare_texts(chunks)

    print("有效 chunk 数量：", len(valid_chunks))

    embeddings = embed_texts(texts, MODEL_NAME)

    print("embedding shape:", embeddings.shape)

    save_outputs(valid_chunks, embeddings, MODEL_NAME)

    print("embedding 已保存到：", EMBEDDINGS_PATH)
    print("chunk 信息已保存到：", CHUNKS_OUTPUT_PATH)
    print("manifest 已保存到：", MANIFEST_PATH)


if __name__ == "__main__":
    main()