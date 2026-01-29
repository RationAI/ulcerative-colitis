from kube_jobs import submit_job


submit_job(
    job_name="ulcerative-colitis-tile-masks-...",
    username=...,
    public=False,
    cpu=64,
    memory="32Gi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.tile_masks +data=tiled/.../...",
    ],
)
