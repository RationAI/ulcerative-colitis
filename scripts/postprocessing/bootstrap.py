from kube_jobs import submit_job


submit_job(
    job_name="ulcerative-colitis-bootstrap-...",
    username=...,
    public=False,
    cpu=2,
    memory="4Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m postprocessing.bootstrap +experiment=postprocessing/final_bootstrap/...",
    ],
)
