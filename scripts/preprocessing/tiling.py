from kube_jobs import storage, submit_job


COHORT = "ikem"  # "ikem", "ftn", or "knl_patos"
TILE_EXTENT = "224px"  # "224px", "320px", or "75um"

submit_job(
    job_name=f"ulcerative-colitis-tiling-{COHORT.replace('_', '-')}-{TILE_EXTENT}",
    username=...,
    public=False,
    cpu=64,
    memory="128Gi",
    shm="48Gi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active -m preprocessing.tiling +experiment=preprocessing/tiling/{COHORT}_{TILE_EXTENT}",
    ],
    storage=[storage.secure.DATA],
)
