import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from rag import config
from rag.agent import QueryAgentWithContext
from rag.data import fetch_text
from rag.agent import generate_response
from rag.schemas import QueryAgentWithContentResponse
from rag.utils import get_num_tokens, trim


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
        assistant_content=assistant_content,
    )

    results = []
    for row in tqdm(data[:num_samples]):
        # gen response
        query = row["question"]
        context = fetch_text(uri=row["source"])
        if len(context) == 0:
            print(f"Empty context for {row['source']}")
            continue
        response: QueryAgentWithContentResponse = agent(
            query=query, context=context
        )
        print(f"Response: {response}")

        # extract from resp
        answer, score, reasoning = extract_from_response(response=response)
        result = {
            "question": query,
            "source": row["source"],
            "answer": answer,
            "score": score,
            "reasoning": reasoning,
        }
        results.append(result)
    return results


def get_retrieval_score(references: List[dict], responses: List[dict]):
    """
    Compute retrieval score based on the sources of the references and responses.
    """
    matches = np.zeros(len(references))
    for i in range(len(references)):
        reference_source = references[i]["source"].split("#")[0]
        if not reference_source:
            matches[i] = 1
            continue
        for source in responses[i]["sources"]:
            # sections don't have to perfectly match
            if reference_source == source.split("#")[0]:
                matches[i] = 1
                continue
    retrieval_score = np.mean(matches)
    return retrieval_score


def evaluate_responses(
    exp_name: str,
    evaluator: str,
    temperature: float,
    max_context_length: int,
    system_content: str,
    assistant_content: str,
    exp_dir: str,
    ref_fp: str,
    resp_fp: str,
    num_samples: Optional[int] = None,
):
    # Load answers
    with open(ref_fp, "r") as f:
        references = [item for item in json.load(f)][:num_samples]
    with open(resp_fp, "r") as f:
        responses = [item for item in json.load(f)['results']][:num_samples]

    
    # Quality score
    results = []
    context_length = max_context_length - get_num_tokens(
        system_content + assistant_content
    )
    for ref, gen in tqdm(zip(references, responses), total=len(references)):
        assert ref["question"] == gen["question"]
        user_content = trim(
            text=str(
                {
                    "question": gen["question"],
                    "generated_answer": gen["answer"],
                    "reference_answer": ref["answer"],
                }
            ),
            max_content_length=context_length,
        )

        # Generate response
        response = generate_response(
            llm_model_name=evaluator,
            temperature=temperature,
            system_content=system_content,
            assistant_content=assistant_content,
            user_content=user_content,
        )

        # Extract from response
        score, reasoning = (
            response.split("\n", maxsplit=1) if "\n" in response else (0, "")
        )
        result = {
            "question": gen["question"],
            "generated_answer": gen["answer"],
            "reference_answer": ref["answer"],
            "score": float(score),
            "reasoning": reasoning,
            "sources": gen["sources"],
        }
        results.append(result)

    # Save results to file
    evaluator_name = evaluator.split("/")[-1]
    evaluation_fp = Path(
        exp_dir, "evaluations", f"{exp_name}_{evaluator_name}.json"
    )
    evaluation_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": exp_name,
        "evaluator": evaluator,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
        "experiments_dir": str(exp_dir),
        "references_fp": str(ref_fp),
        "responses_fp": str(resp_fp),
    }

    evaluation = {
        "config": config,
        "retrieval_score": get_retrieval_score(references, responses),
        "quality_score": np.mean(
            [
                item["score"]
                for item in results
                if (item["score"] and item["reference_answer"])
            ]
        ),
        "results": results,
    }

    with open(evaluation_fp, "w") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)


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
