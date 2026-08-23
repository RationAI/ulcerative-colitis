from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-patch-statistics-...",
    username=...,
    public=False,
    # Keep in sync with num_cpus in explainability/patch_statistics.py's ray.init().
    cpu=32,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active python -m explainability.patch_statistics",
    ],
    storage=[storage.secure.PROJECTS],
)
