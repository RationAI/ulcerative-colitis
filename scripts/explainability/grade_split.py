from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-grade-split-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/grade_split.py's ray.init(). Same root cause as
    # scripts/explainability/patch_statistics.py/nmf_fit.py (oversized
    # parquet row groups) - mean_pool_patches reads the full patch corpus.
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active python -m explainability.grade_split",
    ],
    storage=[storage.secure.PROJECTS],
)
