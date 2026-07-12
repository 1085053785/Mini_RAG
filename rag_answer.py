from search_chunks import load_embeddings, load_chunks, build_bm25_index, retrieve
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


EMBEDDINGS_PATH = Path(r"D:\Dev\PythonWorkspace\documents\embeddings.npy")
CHUNKS_PATH = Path(r"D:\Dev\PythonWorkspace\documents\chunks_embedded.json")
MODEL_NAME = r"D:\Dev\PythonWorkspace\models\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"
LLM_MODEL_PATH = r"D:\Dev\PythonWorkspace\models\Qwen2.5-0.5B-Instruct"
RERANKER_MODEL_PATH = r"D:\Dev\PythonWorkspace\models\bge-reranker-v2-m3"
MIN_SCORE = 0.45



def build_context(results):
    context = ""
    for i, result in enumerate(results, start=1):
        result_text = result["chunk"]["text"]
        context += f"[{i}]\n{result_text}\n\n"
    return context

def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    model.eval()
    return tokenizer, model, device



def ask_llm(context, query, tokenizer, model, device):
    system_prompt = """
你是一个严格的文档问答系统。
你只能根据用户提供的资料回答问题。
如果资料中有相关内容，可以基于资料做简短归纳。
如果资料完全没有相关内容，才回答：资料中没有提到。
禁止编造事实。
禁止使用资料之外的常识。
只能引用资料中实际存在的片段编号。
本次资料只包含 [1]、[2]、[3]，禁止引用 [4]、[5] 或其他不存在的编号。
只回答一次，不要重复“结论”和“证据”。
"""

    user_prompt = f"""
资料：
{context}

问题：
{query}

请只输出一次，格式如下：

结论：用1到2句话回答问题，并引用资料编号。
证据：说明依据来自哪些资料编号。

不要重复以上格式。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer

def check_answer(answer):
    answer = answer.strip()

    if "资料中没有提到" in answer:
        return "资料中没有提到。"

    return answer


def main():
    tokenizer, model, device = load_llm()
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    model_embed = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_PATH,max_length=1024)
    bm25 = build_bm25_index(chunks)
    
    while True:
        print("=" * 100)
        query = input("请输入查询内容：")
        if query.lower() == "q":
            break

        final_results = retrieve(query=query,embeddings=embeddings,chunks=chunks,embedding_model=model_embed,bm25=bm25,reranker=reranker,vector_top_k=5,bm25_top_k=5,fusion_top_k=5,final_k=3)

        top_score = final_results[0]["rerank_score"]
        if top_score < MIN_SCORE:
            print("回答：")
            print("资料中没有提到。")
            print("原因：检索到的资料相关性较低。")
            print(f"最高相似度：{top_score}")
            continue

        context = build_context(final_results)
        answer = ask_llm(context, query, tokenizer, model, device)
        answer = check_answer(answer)

        print(f"答案：{answer}")
        print("=" * 50)
        print("参考来源：")

        for i, result in enumerate(final_results, start=1):
            chunk = result["chunk"]
            metadata = chunk.get("metadata", {})

            source = metadata.get("source", "未知来源")
            chunk_id = metadata.get("chunk_id", "未知编号")

            print(f"[{i}] 来源：{source}，chunk_id：{chunk_id}，相似度：{result['rerank_score']}")
            print(chunk["text"][:200])
            print()

if __name__ == "__main__":
    main()

