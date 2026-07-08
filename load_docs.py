from pathlib import Path

def load_single_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        doc = {"text": text,
                "metadata": {"source": file_path.name,
                            "file_path": str(file_path),
                            "file_type": file_path.suffix}}
    return doc

def load_docs(folder_path):
    folder = Path(folder_path)
    docs = []
    for file_path in folder.rglob("*"):
        if file_path.suffix not in [".md", ".txt"]:
            continue

        doc = load_single_file(file_path)

        docs.append(doc)

    return docs

if __name__ == "__main__":
    docs = load_docs(r"D:\Dev\PythonWorkspace\documents")
    print("读取到的文档数量：", len(docs))
    print(docs[0]["text"])
    print(docs[0]["text"])
    print(docs[0]["metadata"])
    print(docs[0]["metadata"]["source"])
