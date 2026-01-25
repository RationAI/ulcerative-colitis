from kube_jobs import storage, submit_job


COHORT = "ikem"  # "ikem", "ftn", or "knl_patos"

submit_job(
    job_name=f"ulcerative-colitis-dataset-creation-{COHORT.replace('_', '-')}",
    username=...,
    public=False,
    cpu=2,
    memory="4Gi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/ulcerative-colitis.git workdir",
        "cd workdir",
        "git checkout feature/dataset",
        "uv sync --frozen",
        f"uv run -m preprocessing.create_dataset +data=raw/{COHORT}",
    ],
    storage=[storage.secure.DATA],
)
