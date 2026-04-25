from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-embeddings-...",
    username=...,
    public=False,
    cpu=8,
    memory="32Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.embeddings +dataset=tiled/...",
    ],
    storage=[storage.secure.DATA],
)
