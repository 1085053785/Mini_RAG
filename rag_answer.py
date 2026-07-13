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
    load_llm
)

from memory_manager import (
    add_message,
    format_history,
    get_recent_messages
)

from query_rewrite import (
    infer_topic_from_query,
    needs_rewrite,
    rewrite_query
)

from search_chunks import (
    build_bm25_index,
    load_chunks,
    load_embeddings,
    retrieve
)


def build_context(results):
    """
    将最终检索结果转换成LLM Prompt中的参考资料。
    """
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
    """
    打印最终检索结果的来源、chunk_id和相关性分数。
    """
    print("=" * 50)
    print("参考来源：")

    for i, result in enumerate(
        final_results,
        start=1
    ):
        chunk = result["chunk"]

        metadata = chunk.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "未知来源"
        )

        chunk_id = metadata.get(
            "chunk_id",
            "未知编号"
        )

        score = result["rerank_score"]

        print(
            f"[{i}] 来源：{source}，"
            f"chunk_id：{chunk_id}，"
            f"相关性分数：{score:.4f}"
        )

        print(chunk["text"][:200])
        print()


def main():
    # --------------------------------------------------
    # 1. 加载LLM、Embedding模型、Reranker和知识库
    # --------------------------------------------------
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

    # 只保存成功问答轮次
    messages = []

    # 专门供Query Rewrite和检索使用的状态
    retrieval_state = {
        "last_topic": "",
        "last_successful_query": "",
        "last_successful_retrieval_query": ""
    }

    # --------------------------------------------------
    # 2. 多轮问答主循环
    # --------------------------------------------------
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

        # ----------------------------------------------
        # 2.1 获取最近几轮成功历史
        # ----------------------------------------------
        recent_messages = get_recent_messages(
            messages,
            max_turns=MAX_HISTORY_TURNS
        )

        history_text = format_history(
            recent_messages
        )

        # ----------------------------------------------
        # 2.2 判断当前问题是否依赖历史
        # ----------------------------------------------
        query_requires_history = needs_rewrite(
            query
        )

        if query_requires_history:
            # 当前存在指代，但系统还没有有效主题
            if not retrieval_state["last_topic"]:
                print(
                    "回答：你提到的对象不明确，"
                    "请说明具体指什么。"
                )
                continue

            retrieval_query = rewrite_query(
                query=query,
                last_topic=(
                    retrieval_state[
                        "last_topic"
                    ]
                ),
                last_successful_retrieval_query=(
                    retrieval_state[
                        "last_successful_retrieval_query"
                    ]
                ),
                tokenizer=tokenizer,
                model=model,
                device=device
            )
        else:
            # 独立完整问题不经过Query Rewrite
            retrieval_query = query

        print(f"原始问题：{query}")
        print(f"检索问题：{retrieval_query}")
        print(
            "当前有效主题：",
            retrieval_state["last_topic"]
            or "无"
        )

        # ----------------------------------------------
        # 2.3 混合检索、RRF融合和Rerank
        # ----------------------------------------------
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

        # 防止检索结果为空
        if not final_results:
            print("回答：资料中没有提到。")
            print("原因：没有检索到任何资料。")

            # 检索失败，不保存历史，不更新主题
            continue

        top_score = final_results[0][
            "rerank_score"
        ]

        # ----------------------------------------------
        # 2.4 相关性阈值检查
        # ----------------------------------------------
        if top_score < MIN_SCORE:
            print("回答：资料中没有提到。")
            print(
                "原因：检索到的资料相关性较低。"
            )
            print(
                f"最高相关性分数："
                f"{top_score:.4f}"
            )

            # 检索失败，不保存历史，不更新主题
            continue

        # ----------------------------------------------
        # 2.5 构造参考资料并调用LLM回答
        # ----------------------------------------------
        context = build_context(
            final_results
        )

        answer = ask_llm(
            context=context,
            history_text=history_text,
            query=query,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        answer = check_answer(answer)

        print(f"答案：{answer}")

        # 最终模型拒答时，也不更新成功状态
        if answer == "资料中没有提到。":
            continue

        # ----------------------------------------------
        # 2.6 保存成功对话
        # ----------------------------------------------
        add_message(
            messages,
            "user",
            query
        )

        add_message(
            messages,
            "assistant",
            answer
        )

        # ----------------------------------------------
        # 2.7 更新最近一次成功检索状态
        # ----------------------------------------------
        retrieval_state[
            "last_successful_query"
        ] = query

        retrieval_state[
            "last_successful_retrieval_query"
        ] = retrieval_query

        # 独立的新问题会开启或切换主题
        # 追问继续沿用原来的主题
        if not query_requires_history:
            retrieval_state["last_topic"] = (
                infer_topic_from_query(query)
            )

        print(
            "更新后的有效主题：",
            retrieval_state["last_topic"]
        )

        # ----------------------------------------------
        # 2.8 显示参考资料
        # ----------------------------------------------
        print_sources(final_results)


if __name__ == "__main__":
    main()