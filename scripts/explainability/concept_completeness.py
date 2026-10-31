from kube_jobs import storage, submit_job


# Experiment under configs/experiment/explainability/concept_completeness/ -
# "<dictionary>/<evaluation>", e.g. "tree_min_gain_0.005/test_preliminary".
# Dictionaries: configs/explainability/dictionary/ (mlflow URI); evaluations:
# configs/explainability/evaluation/ (split + institutions). One job each.
experiment = ...
# Hydra overrides applied on top of the experiment, space-separated, e.g.
# "dictionary_uri=mlflow-artifacts:/86/<run_id>/artifacts"; "" for none.
# Put a short tag for them in the job name below.
overrides = ...

submit_job(
    job_name=f"uc-completeness-{experiment.replace('/', '-').replace('_', '-')}-...",
    username=...,
    public=False,
    # Streams the full embeddings corpus once - keep cpu in sync with
    # num_cpus in explainability/concept_completeness.py's ray.init().
    cpu=8,
    memory="40Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.concept_completeness "
        f"+experiment=explainability/concept_completeness/{experiment} {overrides}",
    ],
    storage=[storage.secure.PROJECTS],
)
