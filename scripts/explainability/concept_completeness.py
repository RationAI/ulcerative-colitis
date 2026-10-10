from kube_jobs import storage, submit_job


# Output dir of the nmf_fit.py / nmf_tree.py run to check - no default in
# configs/explainability/concept_completeness.yaml, one job per fit.
w_dir = ...

submit_job(
    job_name="ulcerative-colitis-concept-completeness-...",
    username=...,
    public=False,
    # Streams the full embeddings corpus once - keep cpu in sync with
    # num_cpus in explainability/concept_completeness.py's ray.init().
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.concept_completeness w_dir={w_dir}",
    ],
    storage=[storage.secure.PROJECTS],
)
