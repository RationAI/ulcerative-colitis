from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-tile-masks-...",
    username=...,
    public=False,
    cpu=64,
    memory="32Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.tile_masks +dataset=tiled/.../...",
    ],
    storage=[storage.secure.Data],
)
