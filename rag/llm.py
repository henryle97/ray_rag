from typing import List

from openai import OpenAI

import rag.config as config


def get_client():
    # print(f"API URL: {config.LLM_API_URL}")
    client = OpenAI(
        base_url=config.LLM_API_URL,
        api_key=config.LLM_API_KEY,
    )
    return client


def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content


def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].message.content


def send_request(
    model_name: str,
    messages: List[str],
    max_tokens: int = None,
    temperature: float = 0.0,
    stream: bool = False,
):
    client = get_client()
    # print("Calling api...")
    chat_completion = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        messages=messages,
        # For SS llm model
        top_p=0.1,
        extra_body={
            "top_k": 5,
            "repetition_penalty": 1.2,
            "stop_token_ids": [128001, 128009],
        },
    )

    return prepare_response(chat_completion, stream)


if __name__ == "__main__":
    query = "What is ray?"
    # config.LLM_API_URL = "http://107.120.94.21:9113/v1"

    result = send_request(
        # model_name=config.LLM_MODEL_NAME,
        model_name=config.LLM_MODEL_NAME,
        messages=[
            # {"role": "user", "content": query},
            {
                "role": "system",
                "content": "Bạn là Secura, một trợ lý ảo hữu ích chịu trách nhiệm trả lời các câu hỏi liên quan đến bảo mật do SEV và SDSRV phát triển. Trả lời câu hỏi của người dùng ngắn gọn, súc tích, lịch sự và trực tiếp. Dùng tiếng Việt để trả lời.",
            },
            {"role": "user", "content": "Ai tạo ra bạn?"},
        ],
        stream=False,
        max_tokens=1000,
        temperature=0.0,
    )
    print(result)
