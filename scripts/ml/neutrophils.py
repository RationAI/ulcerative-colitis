from kube_jobs import submit_job


submit_job(
    job_name="ulcerative-colitis-ml-neutrophils-...",
    username=...,
    public=False,
    cpu=4,
    memory="8Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m ml.neutrophils +experiment=ml/experiment_vi_neutrophil_detection/run",
    ],
)
