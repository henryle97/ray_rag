import tiktoken


def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
