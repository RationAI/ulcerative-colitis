from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-location-metrics-...",
    username=...,
    public=False,
    cpu=2,
    memory="4Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m postprocessing.location_metrics +experiment=postprocessing/final_location_metrics/ikem",
    ],
    storage=[storage.secure.DATA],
)
