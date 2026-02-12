from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-tiling-...",
    username=...,
    public=False,
    cpu=64,
    memory="128Gi",
    shm="48Gi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.tiling +experiment=preprocessing/tiling/...",
    ],
    storage=[storage.secure.DATA],
)
