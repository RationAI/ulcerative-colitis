from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-quality-control-...",
    username=...,
    public=False,
    cpu=2,
    memory="4Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.quality_control +dataset=processed/...",
    ],
    storage=[storage.secure.DATA],
)
