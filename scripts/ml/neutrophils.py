from kube_jobs import submit_job


submit_job(
    job_name="ulcerative-colitis-neutrophils",
    username=...,
    public=False,
    cpu=4,
    memory="8Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.tissue_masks +data=neutrophils/all",
    ],
)
