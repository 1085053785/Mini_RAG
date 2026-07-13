import torch


# 当前问题以这些词开头时，通常依赖上一轮上下文
REFERENCE_PREFIXES = (
    "它",
    "他",
    "她",
    "这",
    "那"
)

# 当前问题包含这些表达时，通常需要Query Rewrite
REFERENCE_PHRASES = (
    "这个",
    "那个",
    "这种",
    "这样",
    "上述",
    "前者",
    "后者",
    "第一个",
    "第二个",
    "第三个",
    "该方法",
    "该系统",
    "该模型",
    "该功能"
)

# 用于判断LLM输出是否仍然像一个问题
QUESTION_MARKERS = (
    "什么",
    "怎么",
    "如何",
    "为什么",
    "哪些",
    "是否",
    "能否",
    "作用",
    "功能",
    "优点",
    "缺点",
    "区别",
    "应用"
)


def needs_rewrite(query):
    """
    判断当前问题是否依赖上一轮历史。

    例如：
    裸眼3D有什么作用 -> False
    它有什么作用 -> True
    第二个是什么 -> True
    """
    query = query.strip()

    if query.startswith(REFERENCE_PREFIXES):
        return True

    return any(
        phrase in query
        for phrase in REFERENCE_PHRASES
    )


def contains_reference_word(text):
    """
    判断改写结果中是否仍存在未解决的指代词。
    """
    text = text.strip()

    if text.startswith(REFERENCE_PREFIXES):
        return True

    return any(
        phrase in text
        for phrase in REFERENCE_PHRASES
    )


def looks_like_question(text):
    """
    判断LLM输出是否像一个问题。

    主要防止LLM直接输出答案，例如：
    “它可以显示三维图像。”
    """
    text = text.strip()

    if text.endswith(("？", "?")):
        return True

    return any(
        marker in text
        for marker in QUESTION_MARKERS
    )


def infer_topic_from_query(query):
    """
    从独立问题中粗略提取主题。

    示例：
    什么是BM25 -> BM25
    BM25有什么优点 -> BM25
    裸眼3D -> 裸眼3D

    这是规则版主题提取，后续可以替换为LLM实体提取。
    """
    query = query.strip().strip("。？！? ")

    if query.startswith("什么是"):
        topic = query[len("什么是"):].strip()

        if topic:
            return topic

    split_markers = (
        "是什么",
        "能做什么",
        "能干什么",
        "可以做什么",
        "有什么作用",
        "有哪些作用",
        "有哪些功能",
        "有什么功能",
        "有什么优点",
        "有什么缺点",
        "有什么区别",
        "如何实现",
        "怎么实现"
    )

    for marker in split_markers:
        if marker in query:
            topic = query.split(marker, 1)[0].strip()

            if topic:
                return topic

    return query


def clean_rewritten_query(text):
    """
    清理LLM可能附加的前缀、多余解释和空行。
    """
    text = text.strip()

    prefixes = (
        "独立问题：",
        "改写后的独立问题：",
        "改写问题：",
        "检索问题：",
        "输出："
    )

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        text = lines[0]

    return text.strip()


def validate_rewrite(
    original_query,
    rewritten_query
):
    """
    校验LLM生成的改写结果是否可用。
    """
    if not rewritten_query:
        return False

    # 需要改写的问题却被原样返回，视为失败
    if rewritten_query == original_query:
        return False

    # 改写结果仍然包含没有消除的指代词
    if contains_reference_word(rewritten_query):
        return False

    # 防止模型直接输出答案
    if not looks_like_question(rewritten_query):
        return False

    # 防止模型输出异常长文本
    if len(rewritten_query) > 100:
        return False

    return True


def fallback_rewrite(
    query,
    last_topic,
    last_successful_retrieval_query
):
    """
    LLM改写失败后的规则兜底。

    简单指代：
    它能做什么 + 裸眼3D
    -> 裸眼3D能做什么

    复杂指代：
    第二个是什么
    -> 拼接上一次成功检索问题与当前追问
    """
    replaceable_words = (
        "该方法",
        "该系统",
        "该模型",
        "该功能",
        "这个",
        "那个",
        "这种",
        "它",
        "他",
        "她"
    )

    if last_topic:
        for word in replaceable_words:
            if word in query:
                return query.replace(
                    word,
                    last_topic,
                    1
                )

    if last_successful_retrieval_query:
        return (
            f"{last_successful_retrieval_query}。"
            f"当前追问：{query}"
        )

    return query


def rewrite_query(
    query,
    last_topic,
    last_successful_retrieval_query,
    tokenizer,
    model,
    device
):
    """
    使用LLM把依赖上下文的问题改写成独立检索问题。

    如果LLM输出不合格，则自动使用规则兜底。
    """
    system_prompt = """
你是知识库检索系统中的查询改写器。

你的唯一任务是将用户当前的追问改写成一个独立、完整、
脱离历史也能够理解的检索问题。

必须遵守以下规则：

1. 使用“当前有效主题”和“上一次成功检索问题”解析指代。
2. 必须将“它、他、她、这个、那个、这种”等指代词替换为明确对象。
3. 不要回答用户的问题。
4. 输出必须是一个问题，而不是问题的答案。
5. 不得添加输入信息中不存在的新事实。
6. 只输出一个改写后的问题。
7. 不要输出解释、标题、分析或前缀。

示例：

当前有效主题：
BM25

上一次成功检索问题：
什么是BM25？

当前问题：
它有什么优点？

输出：
BM25有什么优点？
""".strip()

    user_prompt = f"""
当前有效主题：
{last_topic}

上一次成功检索问题：
{last_successful_retrieval_query}

当前问题：
{query}

改写后的独立问题：
""".strip()

    prompt_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        prompt_messages,
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
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    new_tokens = output_ids[0][
        input_length:
    ]

    rewritten_query = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    rewritten_query = clean_rewritten_query(
        rewritten_query
    )

    if validate_rewrite(
        original_query=query,
        rewritten_query=rewritten_query
    ):
        print("Query Rewrite方式：LLM改写")
        return rewritten_query

    fallback_query = fallback_rewrite(
        query=query,
        last_topic=last_topic,
        last_successful_retrieval_query=(
            last_successful_retrieval_query
        )
    )

    print("Query Rewrite方式：规则兜底")
    print(f"LLM错误输出：{rewritten_query}")

    return fallback_query