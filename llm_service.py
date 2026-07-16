import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from config import LLM_MODEL_PATH


def load_llm():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True
    )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    model.eval()

    return tokenizer, model, device


def format_history(messages):
    """
    将结构化消息转换成适合放入 Prompt 的历史文本。
    """
    history_lines = []

    for message in messages:
        if message["role"] == "user":
            role_name = "用户"
        elif message["role"] == "assistant":
            role_name = "助手"
        else:
            role_name = message["role"]

        history_lines.append(
            f"{role_name}：{message['content']}"
        )

    return "\n".join(history_lines)


def summarize_messages(
    old_summary,
    old_messages,
    tokenizer,
    model,
    device
):
    """
    使用旧摘要和本次移出短期窗口的消息，
    生成新的滚动摘要。
    """
    if not old_messages:
        return old_summary

    old_history = format_history(
        old_messages
    )

    user_prompt = f"""
已有对话摘要：
{old_summary if old_summary else "暂无"}

本次新增的旧对话：
{old_history}

请将已有摘要和新增旧对话合并成新的对话摘要。

要求：
1. 保留主要讨论对象。
2. 保留用户的重要需求、目标和约束。
3. 保留已经确定的事实和结论。
4. 保留尚未解决的问题。
5. 删除重复内容、寒暄和无关细节。
6. 不得添加对话中不存在的信息。
7. 使用简洁、明确的中文陈述。
8. 只输出更新后的摘要，不要解释。
""".strip()

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "你负责维护对话摘要。"
                "你的任务是压缩和合并历史对话，"
                "不能回答问题，也不能添加原对话中不存在的信息。"
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

    input_length = inputs[
        "input_ids"
    ].shape[1]

    new_tokens = output_ids[0][
        input_length:
    ]

    new_summary = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return new_summary.strip()


def ask_llm(
    context,
    history_text,
    summary,
    query,
    tokenizer,
    model,
    device
):
    if not history_text:
        history_text = "无历史对话"

    if not summary:
        summary = "无历史摘要"

    system_prompt = """
你是一个严格的文档问答系统。

1. 参考资料是回答事实问题的唯一依据。
2. 如果资料中有相关内容，可以基于资料做简短归纳。
3. 历史对话和历史摘要只用于理解上下文、指代和用户意图，不能作为事实依据。
4. 不允许编造参考资料中没有出现的信息。
5. 如果参考资料不足以回答问题，请回答“资料中没有提到”。
6. 只能引用资料部分实际存在的编号，禁止引用不存在的编号。
7. 回答应直接、清晰，不要重复参考资料全文。
""".strip()

    user_prompt = f"""
历史摘要：
{summary}

最近对话：
{history_text}

参考资料：
{context}

当前问题：
{query}

请只输出一次，格式如下：

结论：用1到2句话回答问题，并引用资料编号。
证据：说明依据来自哪些资料编号。

不要重复以上格式。
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
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.15,
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

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


def check_answer(answer):
    """
    统一模型拒答文本，避免带格式的拒答被误判为成功回答。
    """
    answer = answer.strip()

    if "资料中没有提到" in answer:
        return "资料中没有提到。"

    return answer
