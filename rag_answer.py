from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer
)

from config import (
    BM25_TOP_K,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_PATH,
    FINAL_TOP_K,
    FUSION_TOP_K,
    MAX_HISTORY_TURNS,
    MIN_SCORE,
    RERANKER_MODEL_PATH,
    VECTOR_TOP_K
)

from llm_service import (
    ask_llm,
    check_answer,
    format_history,
    load_llm,
    summarize_messages
)

from memory_manager import (
    add_message,
    split_messages
)

from query_rewrite import (
    decide_query_action,
    infer_topic_from_query,
    rewrite_query
)

from search_chunks import (
    build_bm25_index,
    load_chunks,
    load_embeddings,
    retrieve
)


def build_context(results):
    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):
        result_text = result[
            "chunk"
        ]["text"]

        context_parts.append(
            f"[{i}]\n{result_text}"
        )

    return "\n\n".join(context_parts)


def print_sources(final_results):
    print("=" * 50)
    print("参考来源：")

    for i, result in enumerate(
        final_results,
        start=1
    ):
        chunk = result["chunk"]
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "未知来源")
        chunk_id = metadata.get("chunk_id", "未知编号")
        score = result["rerank_score"]

        print(
            f"[{i}] 来源：{source}，"
            f"chunk_id：{chunk_id}，"
            f"相关性分数：{score:.4f}"
        )
        print(chunk["text"][:200])
        print()


def main():
    tokenizer, model, device = load_llm()

    embeddings = load_embeddings(
        EMBEDDINGS_PATH
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    model_embed = SentenceTransformer(
        EMBEDDING_MODEL_PATH
    )

    reranker = CrossEncoder(
        RERANKER_MODEL_PATH,
        max_length=1024
    )

    bm25 = build_bm25_index(chunks)

    messages = []
    conversation_summary = ""

    retrieval_state = {
        "last_topic": "",
        "last_successful_query": "",
        "last_successful_retrieval_query": ""
    }

    while True:
        print("=" * 100)

        query = input(
            "请输入查询内容："
        ).strip()

        if query.lower() == "q":
            break

        if not query:
            print("查询内容不能为空。")
            continue

        history_text = format_history(
            messages
        )
        summary = conversation_summary

        query_action = decide_query_action(
            query=query,
            history_text=history_text,
            conversation_summary=conversation_summary,
            last_topic=retrieval_state["last_topic"],
            last_successful_retrieval_query=(
                retrieval_state[
                    "last_successful_retrieval_query"
                ]
            )
        )

        if query_action == "clarify":
            print(
                "回答：你提到的对象不明确，"
                "请说明具体指什么。"
            )
            continue

        if query_action == "rewrite":
            retrieval_query = rewrite_query(
                query=query,
                last_topic=retrieval_state["last_topic"],
                last_successful_retrieval_query=(
                    retrieval_state[
                        "last_successful_retrieval_query"
                    ]
                ),
                history_text=history_text,
                conversation_summary=conversation_summary,
                tokenizer=tokenizer,
                model=model,
                device=device
            )
        else:
            retrieval_query = query

        print(f"Query路由：{query_action}")
        print(f"原始问题：{query}")
        print(f"检索问题：{retrieval_query}")
        print(
            "当前有效主题：",
            retrieval_state["last_topic"] or "无"
        )

        final_results = retrieve(
            query=retrieval_query,
            embeddings=embeddings,
            chunks=chunks,
            embedding_model=model_embed,
            bm25=bm25,
            reranker=reranker,
            vector_top_k=VECTOR_TOP_K,
            bm25_top_k=BM25_TOP_K,
            fusion_top_k=FUSION_TOP_K,
            final_k=FINAL_TOP_K
        )

        if not final_results:
            print("回答：资料中没有提到。")
            print("原因：没有检索到任何资料。")
            continue

        top_score = final_results[0][
            "rerank_score"
        ]

        if top_score < MIN_SCORE:
            print("回答：资料中没有提到。")
            print("原因：检索到的资料相关性较低。")
            print(f"最高相关性分数：{top_score:.4f}")
            continue

        context = build_context(
            final_results
        )

        answer = ask_llm(
            context=context,
            history_text=history_text,
            summary=summary,
            query=query,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        answer = check_answer(answer)
        print(f"答案：{answer}")

        if answer == "资料中没有提到。":
            continue

        add_message(messages, "user", query)
        add_message(messages, "assistant", answer)

        old_messages, recent_messages = split_messages(
            messages,
            max_turns=MAX_HISTORY_TURNS
        )

        if old_messages:
            conversation_summary = summarize_messages(
                old_summary=conversation_summary,
                old_messages=old_messages,
                tokenizer=tokenizer,
                model=model,
                device=device
            )
            messages[:] = recent_messages

        retrieval_state["last_successful_query"] = query
        retrieval_state[
            "last_successful_retrieval_query"
        ] = retrieval_query

        if query_action == "direct":
            new_topic = infer_topic_from_query(
                retrieval_query
            )
            if new_topic:
                retrieval_state["last_topic"] = new_topic

        print(
            "更新后的有效主题：",
            retrieval_state["last_topic"] or "无"
        )

        print_sources(final_results)


if __name__ == "__main__":
    main()
