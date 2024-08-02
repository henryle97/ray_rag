import rag.config as config
from rag.generate import QueryAgent

if __name__ == "__main__":
    agent = QueryAgent(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        llm_model_name=config.LLM_MODEL_NAME,
        max_context_length=config.MAX_CONTEXT_LENGTHS[config.LLM_MODEL_NAME],
        system_content=config.SYSTEM_CONTENT,
    )

    query = "What is ray?"
    result = agent(
        query=query,
        stream=False,
    )
    print(result)
