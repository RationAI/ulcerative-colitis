from kube_jobs import storage, submit_job


submit_job(
    job_name=f"ulcerative-colitis-embeddings-xai-...",
    username=...,
    public=False,
    cpu=16,
    memory="128Gi",
    shm="64Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.embeddings_xai +dataset=tiled/... +experiment=preprocessing/embeddings_xai/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
