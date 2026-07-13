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


def ask_llm(
    context,
    history_text,
    query,
    tokenizer,
    model,
    device
):
    if not history_text:
        history_text = "无历史对话"

    system_prompt = """
你是一个严格的文档问答系统。

1. 参考资料是回答事实问题的唯一依据。
2. 如果资料中有相关内容，可以基于资料做简短归纳。
3. 历史对话只用于理解上下文、指代和用户意图，不能作为事实依据。
4. 不允许编造参考资料中没有出现的信息。
5. 如果参考资料不足以回答问题，请回答“资料中没有提到”。
6. 只能引用资料部分实际存在的编号，禁止引用不存在的编号。
7. 回答应直接、清晰，不要重复参考资料全文。
""".strip()

    user_prompt = f"""
历史对话：
{history_text}

资料：
{context}

问题：
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
    answer = answer.strip()

    no_answer_phrases = {
        "资料中没有提到",
        "资料中没有提到。",
        "结论：资料中没有提到",
        "结论：资料中没有提到。"
    }

    if answer in no_answer_phrases:
        return "资料中没有提到。"

    return answer