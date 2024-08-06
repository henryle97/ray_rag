from pathlib import Path
from typing import Optional

from rag import config
from rag.evaluation.evaluate import evaluate_responses
from rag.generate import generate_responses
from rag.schemas import HyperParameter


def run_experiment(
    exp_name: str,
    hyperparams: HyperParameter,
    docs_dir: str,
    exp_dir: str,
    ref_fp: str,
    system_content: str = config.EXPERIMENT_SYSTEM_CONTENT,
    num_samples: Optional[int] = None,
):
    """
    Generate responses and evaluate them.
    """
    # generate response
    resp_fp = str(Path(exp_dir) / "responses" / f"{exp_name}.json")
    generate_responses(
        exp_name=exp_name,
        hyperparams=hyperparams,
        docs_dir=docs_dir,
        exp_dir=exp_dir,
        ref_fp=ref_fp,
        system_content=system_content,
        assistant_content="",
        num_samples=num_samples,
        resp_fp=resp_fp
    )
    
    # Evaluate responses
    evaluation_system_content = """
        Your job is to rate the quality of our generated answer {generated_answer}
        given a query {query} and a reference answer {reference_answer}.
        Your score has to be between 1 and 5.
        You must return your response in a line with only the score.
        Do not return answers in any other format.
        On a separate line provide your reasoning for the score as well.
        """
    evaluate_responses(
        exp_name=exp_name,
        evaluator=hyperparams.evaluator,
        temperature=hyperparams.temperature,
        max_context_length=hyperparams.max_context_length,
        system_content=evaluation_system_content,
        assistant_content="",
        exp_dir=exp_dir,
        ref_fp=ref_fp,
        resp_fp=resp_fp,
        num_samples=num_samples
    )


if __name__ == "__main__":
    hyperparams = HyperParameter()
    run_experiment(
        exp_name="without-context",
        hyperparams=hyperparams,
        docs_dir=config.DOCS_DIR,
        exp_dir=config.EXPERIMENTS_DIR,
        ref_fp=config.REFERENCES_FILE_PATH,
        system_content=config.EXPERIMENT_SYSTEM_CONTENT,
        num_samples=3,
    )
