from load_docs import load_docs
import json


def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line == "":
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

def split_doc(doc, max_chars=500, overlap_chars=100):
    text = doc["text"]
    metadata = doc["metadata"]

    text = clean_text(text)

    paragraphs = text.split("\n\n")
    if len(paragraphs) == 1:
        paragraphs = text.split("\n")

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph == "":
            continue

        # 如果当前 chunk 加上这个 paragraph 还没有超过最大长度，就继续合并
        if len(current_chunk) + len(paragraph) + 2 <= max_chars:
            if current_chunk == "":
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph

        # 如果加上这个 paragraph 会超过最大长度，就先保存当前 chunk
        else:
            if current_chunk:
                chunk = {
                    "text": current_chunk,
                    "metadata": {
                        "source": metadata["source"],
                        "file_path": metadata["file_path"],
                        "file_type": metadata["file_type"],
                        "chunk_id": f"{metadata['source']}_chunk_{chunk_index}"
                    }
                }
                chunks.append(chunk)
                chunk_index += 1

                # overlap：取上一个 chunk 结尾的一部分，放到新 chunk 开头
                if overlap_chars > 0:
                    overlap_text = current_chunk[-overlap_chars:]
                else:
                    overlap_text = ""

                if overlap_text:
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk = paragraph

    # 最后一个 chunk 也要保存
    if current_chunk:
        chunk = {
            "text": current_chunk,
            "metadata": {
                "source": metadata["source"],
                "file_path": metadata["file_path"],
                "file_type": metadata["file_type"],
                "chunk_id": f"{metadata['source']}_chunk_{chunk_index}"
            }
        }
        chunks.append(chunk)

    return chunks

def split_docs(docs):
    all_chunks = []
    for doc in docs:
        chunks = split_doc(doc)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    docs = load_docs(r"D:\Dev\PythonWorkspace")
    chunks = split_docs(docs)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    for chunk in chunks[:10]:
        print("chunk_id:", chunk["metadata"]["chunk_id"])
        print("length:", len(chunk["text"]))
        print(chunk["text"])
        print("-" * 50)