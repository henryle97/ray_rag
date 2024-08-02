from qdrant_client.conversions import common_types as types
from rag.llm import send_request


def generate_response(
    llm_model_name: str,
    max_tokens: int = None,
    temperature: float = 0.0,
    stream: bool = False,
    system_content: str = "",
    assistant_content: str = "",
    user_content: str = "",
) -> str:
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", system_content),
            ("assistant", assistant_content),
            ("user", user_content),
        ]
        if content
    ]
    print("Messages: ", messages)
    return send_request(
        model_name=llm_model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )

