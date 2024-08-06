import ray

from rag import config
from rag.generate import generate_response


def load_chunks_from(parquet_path: str):
    data = ray.data.read_parquet(parquet_path)
    return data


if __name__ == "__main__":

    num_questions = 3
    system_content = f"""
    Create {num_questions} questions using only the context provided.
    Write the answer to that question below question, using only the context provided.
    Separate each question/answer pair by a newline.
    """

    chunks_ds = load_chunks_from(parquet_path=config.PARQUET_FILE)

    # get a sample
    sample = chunks_ds.take(1)
    print(sample)

    synthetic_data = []
    row_cnt = 0
    for row in chunks_ds.iter_rows():
        response = generate_response(
            llm_model_name=config.LLM_MODEL_NAME,
            temperature=config.TEMPERATURE,
            stream=False,
            system_content=system_content,
            user_content=f"context: {row['text']}",
        )
        print(response)
        response = response.replace(
            "Here are three questions based on the given context:\n\n", ""
        )
        response = response.replace("\nAnswer:", "")
        print(f"After replace:\n{response}")
        entries = response.split("\n\n")
        print(f"Entries: {entries}")
        for entry in entries:
            items = entry.split("\n")
            if len(items) != 2:
                print(f"Skipping entry: {entry}")
                continue
            question, answer = items[0], items[1:]
            synthetic_data.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": row["source"],
                }
            )

        row_cnt += 1
        if row_cnt > 2:
            break
    print(f"Synthetic data:\n{synthetic_data}")
