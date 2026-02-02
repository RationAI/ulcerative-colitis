from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-neutrophils-...",
    username=...,
    public=False,
    cpu=16,
    memory="32Gi",
    shm="32Gi",
    gpu="mig-2g.20gb",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.neutrophils +data=tiled/.../0_320",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
