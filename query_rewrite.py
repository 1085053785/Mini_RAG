import re

import torch


HISTORY_REFERENCE_PHRASES = (
    "之前说的",
    "之前提到的",
    "刚才说的",
    "刚才提到的",
    "前面说的",
    "前面提到的",
    "上面说的",
    "上面提到的",
    "上述",
    "前述",
)

GENERIC_REFERENCE_PATTERNS = (
    r"(^|[，。！？；：,\s])它(?=[\u4e00-\u9fffA-Za-z0-9]|[呢吗的，。！？；：,\s]|$)",
    r"(^|[，。！？；：,\s])他(?=[\u4e00-\u9fffA-Za-z0-9]|[呢吗的，。！？；：,\s]|$)",
    r"(^|[，。！？；：,\s])她(?=[\u4e00-\u9fffA-Za-z0-9]|[呢吗的，。！？；：,\s]|$)",
    r"(^|[，。！？；：,\s])这个(?=[呢吗，。！？；：,\s]|$)",
    r"(^|[，。！？；：,\s])那个(?=[呢吗，。！？；：,\s]|$)",
    r"(^|[，。！？；：,\s])这些(?=[呢吗，。！？；：,\s]|$)",
    r"(^|[，。！？；：,\s])那些(?=[呢吗，。！？；：,\s]|$)",
    r"(这个|那个|该)(系统|功能|技术|方法|模型|任务|设备|项目|对象|问题)",
)

FOLLOW_UP_PREFIXES = (
    "比如说",
    "比如",
    "例如",
    "那",
    "那么",
    "还有",
    "具体",
    "继续",
    "然后",
    "所以",
    "再说说",
)

SHORT_FOLLOW_UP_PATTERNS = (
    r"^为什么[呢吗]?[？?]?$",
    r"^怎么办[呢吗]?[？?]?$",
    r"^怎么做[呢吗]?[？?]?$",
    r"^有什么要求[呢吗]?[？?]?$",
    r"^有什么作用[呢吗]?[？?]?$",
    r"^有哪些[呢吗]?[？?]?$",
    r"^有什么[呢吗]?[？?]?$",
    r"^适合谁[呢吗]?[？?]?$",
    r"^能做什么[呢吗]?[？?]?$",
    r"^能干什么[呢吗]?[？?]?$",
    r"^区别呢[？?]?$",
    r"^优缺点呢[？?]?$",
    r"^原因呢[？?]?$",
    r"^举个例子[呢吗]?[？?]?$",
)

QUESTION_INTENT_WORDS = (
    "什么",
    "哪些",
    "为什么",
    "怎么",
    "如何",
    "是否",
    "能否",
    "作用",
    "功能",
    "要求",
    "证据",
    "区别",
    "优点",
    "缺点",
    "适合",
    "训练",
)


def _normalize_text(text):
    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def _has_context(
    history_text="",
    conversation_summary="",
    last_topic="",
    last_successful_retrieval_query=""
):
    return bool(
        _normalize_text(history_text)
        or _normalize_text(conversation_summary)
        or _normalize_text(last_topic)
        or _normalize_text(last_successful_retrieval_query)
    )


def _contains_history_reference(query):
    return any(
        phrase in query
        for phrase in HISTORY_REFERENCE_PHRASES
    )


def _contains_generic_reference(query):
    return any(
        re.search(pattern, query)
        for pattern in GENERIC_REFERENCE_PATTERNS
    )


def _is_follow_up(query):
    if any(
        re.search(pattern, query)
        for pattern in SHORT_FOLLOW_UP_PATTERNS
    ):
        return True

    for prefix in sorted(
        FOLLOW_UP_PREFIXES,
        key=len,
        reverse=True
    ):
        if query.startswith(prefix):
            remainder = query[len(prefix):].strip(
                "，,：: "
            )

            # “那么RRF融合有什么作用”已经带有明确主题，
            # 不应仅因为开头有“那么”就强制改写。
            if infer_topic_from_query(remainder):
                return False

            return True

    return False


def decide_query_action(
    query,
    history_text="",
    conversation_summary="",
    last_topic="",
    last_successful_retrieval_query=""
):
    """
    返回以下三种动作之一：

    direct  ：问题本身完整，直接检索。
    rewrite ：问题依赖历史，且存在可用于改写的上下文。
    clarify ：问题依赖历史，但没有可用上下文，应要求用户澄清。
    """
    query = _normalize_text(query)

    if not query:
        return "clarify"

    depends_on_history = (
        _contains_history_reference(query)
        or _contains_generic_reference(query)
        or _is_follow_up(query)
    )

    if not depends_on_history:
        return "direct"

    if _has_context(
        history_text=history_text,
        conversation_summary=conversation_summary,
        last_topic=last_topic,
        last_successful_retrieval_query=(
            last_successful_retrieval_query
        )
    ):
        return "rewrite"

    return "clarify"


def needs_rewrite(
    query,
    history_text="",
    conversation_summary="",
    last_topic="",
    last_successful_retrieval_query=""
):
    """
    兼容原有主流程。

    rewrite 和 clarify 都表示当前问题依赖历史；
    主流程可在没有上下文时输出澄清提示。
    """
    action = decide_query_action(
        query=query,
        history_text=history_text,
        conversation_summary=conversation_summary,
        last_topic=last_topic,
        last_successful_retrieval_query=(
            last_successful_retrieval_query
        )
    )

    return action != "direct"


def _clean_topic(topic):
    topic = _normalize_text(topic)

    topic = topic.strip(
        "，。！？；：,.!?;:、“”‘’\"' "
    )

    topic = re.sub(
        r"^(请问|请介绍一下|介绍一下|说一下|讲一下|"
        r"我想了解|我想知道|我想问|关于|针对)",
        "",
        topic
    ).strip()

    if not topic:
        return ""

    if topic in {
        "这个",
        "那个",
        "它",
        "他",
        "她",
        "这个系统",
        "那个系统",
        "该系统",
        "这个问题",
    }:
        return ""

    if len(topic) > 32:
        return ""

    return topic


def infer_topic_from_query(query):
    """
    从独立问题中提取简短主题。
    """
    query = _normalize_text(query)

    if not query:
        return ""

    patterns = (
        r"(?:我想|我要|希望)?(?:用|使用)(.+?)"
        r"(?:完成|做|实现|进行|开发|构建|来)",

        r"(?:关于|针对)(.+?)"
        r"(?:，|,|是什么|是什么意思|有哪些|有什么|怎么|如何|"
        r"为什么|是否|能否|可以|$)",

        r"^(.+?)的(?:参数|作用|原理|要求|优点|缺点|流程|区别|"
        r"特点|用途|实现|证据)",

        r"^(.+?)(?:是什么|是什么意思|能做什么|能干什么|"
        r"有什么作用|有什么区别|有哪些|有什么要求|"
        r"怎么实现|如何实现|为什么|是否|能否|适合谁|"
        r"包含哪些|使用了哪些|主要服务于)",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            query
        )

        if match:
            topic = _clean_topic(
                match.group(1)
            )

            if topic:
                return topic

    technical_terms = re.findall(
        r"[A-Za-z][A-Za-z0-9_.+-]*(?:\s+[A-Za-z][A-Za-z0-9_.+-]*)*"
        r"|[\u4e00-\u9fff]{1,10}(?:3D|AI)",
        query
    )

    if technical_terms:
        return _clean_topic(
            technical_terms[0]
        )

    return ""


def _remove_follow_up_prefix(query):
    query = query.strip()

    for prefix in sorted(
        FOLLOW_UP_PREFIXES,
        key=len,
        reverse=True
    ):
        if query.startswith(prefix):
            return query[len(prefix):].lstrip(
                "，,：: "
            )

    return query


def _replace_references(query, topic):
    replacements = (
        "该系统",
        "该技术",
        "该方法",
        "该模型",
        "该任务",
        "这个系统",
        "这个技术",
        "这个方法",
        "这个模型",
        "这个任务",
        "这个功能",
        "那个系统",
        "那个功能",
        "这个",
        "那个",
        "它",
        "他",
        "她",
    )

    result = query

    for reference in replacements:
        result = result.replace(
            reference,
            topic
        )

    return result


def rule_based_rewrite(
    query,
    last_topic="",
    last_successful_retrieval_query=""
):
    query = _normalize_text(query)

    topic = _clean_topic(last_topic)

    if not topic:
        topic = infer_topic_from_query(
            last_successful_retrieval_query
        )

    if not topic:
        return query

    # “之前说的三维图像是什么意思”
    history_match = re.search(
        r"(?:之前|刚才|前面|上面)"
        r"(?:说的|提到的)?(.+)",
        query
    )

    if history_match:
        content = history_match.group(
            1
        ).strip("，,：: ")

        content = _replace_references(
            content,
            topic
        )

        if topic not in content:
            return f"{topic}中的{content}"

        return content

    # “那任务五呢 / 那任务五主要训练什么”
    task_match = re.search(
        r"任务([一二三四五六七八九十\d]+)",
        query
    )

    if task_match:
        task_name = (
            f"任务{task_match.group(1)}"
        )

        return (
            f"{topic}中的{task_name}"
            f"主要训练哪些认知功能？"
        )

    if re.search(
        r"(能干什么|能做什么|可以做什么)",
        query
    ):
        return (
            f"{topic}有哪些功能和训练任务？"
        )

    rewritten_query = _remove_follow_up_prefix(
        query
    )

    rewritten_query = _replace_references(
        rewritten_query,
        topic
    )

    if rewritten_query.endswith(
        ("呢？", "呢?")
    ):
        base = rewritten_query[:-2].strip()

        return f"{base}有什么作用？"

    if rewritten_query.endswith("呢"):
        base = rewritten_query[:-1].strip()

        return f"{base}有什么作用？"

    if topic in rewritten_query:
        return rewritten_query

    if rewritten_query.startswith(
        "什么应用要求"
    ):
        return f"{topic}有哪些应用要求？"

    if rewritten_query.startswith(
        "什么要求"
    ):
        return f"{topic}有什么要求？"

    if rewritten_query.startswith(
        (
            "有什么",
            "有哪些",
            "为什么",
            "怎么",
            "如何",
            "是否",
            "能否",
            "可以",
            "适合",
        )
    ):
        return f"{topic}{rewritten_query}"

    return f"关于{topic}，{rewritten_query}"


def _clean_model_output(output):
    output = _normalize_text(output)

    output = re.sub(
        r"^(改写后的问题|改写问题|独立问题|检索问题|结果)"
        r"[：:]\s*",
        "",
        output
    ).strip()

    return output.strip(
        "\"'“”‘’"
    )


def _is_valid_rewrite(
    original_query,
    rewritten_query,
    topic
):
    if not rewritten_query:
        return False

    if len(rewritten_query) > 120:
        return False

    if _contains_history_reference(
        rewritten_query
    ):
        return False

    if _contains_generic_reference(
        rewritten_query
    ):
        return False

    if rewritten_query.endswith(
        ("呢", "呢？", "呢?")
    ):
        return False

    if rewritten_query == original_query:
        return False

    invalid_markers = (
        "无法改写",
        "不需要改写",
        "根据上下文",
        "用户的问题",
        "改写如下",
    )

    if any(
        marker in rewritten_query
        for marker in invalid_markers
    ):
        return False

    if (
        topic
        and topic not in rewritten_query
        and not infer_topic_from_query(
            rewritten_query
        )
    ):
        return False

    if not any(
        word in rewritten_query
        for word in QUESTION_INTENT_WORDS
    ):
        return False

    return True


def rewrite_query(
    query,
    last_topic,
    last_successful_retrieval_query,
    tokenizer,
    model,
    device,
    history_text="",
    conversation_summary=""
):
    query = _normalize_text(query)

    topic = _clean_topic(last_topic)

    if not topic:
        topic = infer_topic_from_query(
            last_successful_retrieval_query
        )

    fallback_query = rule_based_rewrite(
        query=query,
        last_topic=topic,
        last_successful_retrieval_query=(
            last_successful_retrieval_query
        )
    )

    recent_history = (
        history_text[-1500:]
        if history_text
        else "无"
    )

    summary_text = (
        conversation_summary[-800:]
        if conversation_summary
        else "无"
    )

    user_prompt = f"""
请把当前问题改写成一句可以脱离对话历史、直接用于文档检索的完整问题。

要求：
1. 只补全被省略的对象、主题和必要限定词。
2. 优先依据最近对话，其次依据当前主题和上次检索问题，最后参考历史摘要。
3. 保留用户原意，不回答问题，不增加新事实。
4. 删除“之前说的、刚才提到的、比如说、那个、它”等依赖上下文的表达。
5. 输出必须明确说明用户询问的对象和具体意图。
6. 必须原样保留当前问题中已经明确出现的专业词和名词，不要擅自替换同义词。
7. 不能输出“任务五呢”这类仍不完整的追问。
8. 只输出改写后的一个问题，不要解释。

当前主题：
{topic or "无"}

上次成功检索问题：
{last_successful_retrieval_query or "无"}

历史摘要：
{summary_text}

最近对话：
{recent_history}

当前问题：
{query}
""".strip()

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "你是文档检索查询改写器。"
                "你只负责将依赖上下文的问题改写成独立查询，"
                "不能回答问题。"
            )
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

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.10,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs[
            "input_ids"
        ].shape[1]

        new_tokens = output_ids[0][
            input_length:
        ]

        model_output = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        rewritten_query = _clean_model_output(
            model_output
        )

        if _is_valid_rewrite(
            original_query=query,
            rewritten_query=rewritten_query,
            topic=topic
        ):
            print("Query Rewrite方式：LLM")
            return rewritten_query

        print("Query Rewrite方式：规则兜底")
        print(f"LLM错误输出：{rewritten_query}")

        return fallback_query

    except Exception as error:
        print("Query Rewrite方式：规则兜底")
        print(f"LLM改写异常：{error}")

        return fallback_query
