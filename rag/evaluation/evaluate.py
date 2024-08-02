import json
from pathlib import Path
import re
from typing import List

from tqdm import tqdm
from rag import config
from rag.data import fetch_text
from rag.agent import QueryAgentWithContext
from rag.schemas import QueryAgentWithContentResponse

def extract_from_response(response: QueryAgentWithContentResponse):
    # Define regular expressions for extracting values
    answer_pattern = r'"answer"\s*:\s*"([^"]*)"'
    score_pattern = r'"score"\s*:\s*([0-9]+)'
    reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'

    # Extract values using regular expressions
    answer_match = re.search(answer_pattern, response.answer)
    score_match = re.search(score_pattern, response.answer)
    reasoning_match = re.search(reasoning_pattern, response.answer)

    # Convert
    if answer_match and score_match and reasoning_match:
        answer = answer_match.group(1)
        score = float(score_match.group(1))
        reasoning = reasoning_match.group(1)
        return answer, score, reasoning

    return "", "", ""

def get_references(
    data: List[dict], 
    num_samples: int,
    llm_model_name: str,
    temperature: float,
    system_content: str,
    assistant_content: str,
):
    # init agent
    agent = QueryAgentWithContext(
        llm_model_name=llm_model_name,
        temperature=temperature,
        system_content=system_content,
        assistant_content=assistant_content
    )
    
    results = []
    for row in tqdm(data[:num_samples]):
        # gen response 
        query = row["question"]
        context = fetch_text(uri=row["source"])
        if len(context) == 0:
            print(f"Empty context for {row['source']}")
            continue
        response: QueryAgentWithContentResponse = agent(query=query, context=context)
        print(f"Response: {response}")

        # extract from resp
        answer, score, reasoning = extract_from_response(response=response)
        result = {
            "question": query,
            "source": row["source"],
            "answer": answer,
            "score": score,
            "reasoning": reasoning
        }
        results.append(result)
    return results
        

if __name__ == "__main__":
    with open(Path(config.ROOT_DIR, "data/eval-dataset-v1.jsonl"), "r") as f:
        eval_data = [json.loads(item) for item in list(f)]
        
    print(f"Eval data: {eval_data[0]}")

    # Sample
    uri = "https://docs.ray.io/en/master/data/transforming-data.html#configuring-batch-format"
    text = fetch_text(uri=uri)
    print(f"Text content at {uri}: {text[:100]}...")
    
    
    results = get_references(
        data=eval_data,
        llm_model_name=config.LLM_MODEL_NAME,
        num_samples=config.NUM_EVAL_SAMPLES,
        temperature=config.TEMPERATURE,
        system_content=config.EVALUATE_SYSTEM_CONTENT,
        assistant_content=config.EVALUATE_ASSISTANT_CONTENT,
    )
    
    # Save to file
    Path(config.REFERENCES_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.REFERENCES_FILE_PATH, "w") as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False)