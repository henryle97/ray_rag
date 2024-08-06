from typing import Optional

from rag import config
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
    generate_responses(
        exp_name=exp_name,
        hyperparams=hyperparams,
        docs_dir=docs_dir,
        exp_dir=exp_dir,
        ref_fp=ref_fp,
        system_content=system_content,
        assistant_content="",
        num_samples=num_samples,
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
