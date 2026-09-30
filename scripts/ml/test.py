from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-ml-test-...",
    username=...,
    public=False,
    gpu="mig-1g.10gb",
    cpu=10,
    memory="64Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m ml +experiment=ml/.../...",
    ],
    storage=[storage.secure.DATA],
)
