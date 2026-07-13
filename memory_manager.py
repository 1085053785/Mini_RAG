def add_message(messages, role, content):
    messages.append({
        "role": role,
        "content": content
    })


def get_recent_messages(messages, max_turns=3):
    return messages[-max_turns * 2:]


def format_history(messages):
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