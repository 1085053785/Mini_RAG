def add_message(messages, role, content):
    """
    向短期记忆中追加一条消息。
    """
    messages.append({
        "role": role,
        "content": content
    })


def get_recent_messages(messages, max_turns=3):
    """
    返回最近 max_turns 轮对话。

    一轮默认包含一条 user 消息和一条 assistant 消息。
    """
    return messages[-max_turns * 2:]


def split_messages(messages, max_turns=3):
    """
    将消息拆分成：
    1. 需要进入摘要的旧消息
    2. 继续保留在短期记忆中的最近消息
    """
    keep_count = max_turns * 2

    if len(messages) <= keep_count:
        return [], messages

    old_messages = messages[:-keep_count]
    recent_messages = messages[-keep_count:]

    return old_messages, recent_messages
