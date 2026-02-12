from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-embeddings-...",
    username=...,
    public=False,
    cpu=16,
    memory="32Gi",
    gpu="H100",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "export HF_TOKEN=...",
        "uv sync --frozen",
        "uv run -m preprocessing.embeddings +experiment=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
