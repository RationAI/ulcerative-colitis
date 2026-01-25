from kube_jobs import storage, submit_job


COHORT = "ikem"  # "ikem", "ftn", or "knl_patos"

submit_job(
    job_name=f"ulcerative-colitis-tissue-masks-{COHORT.replace('_', '-')}",
    username=...,
    public=False,
    cpu=32,
    memory="128Gi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active -m preprocessing.tissue_masks +data=processed/{COHORT}",
    ],
    storage=[storage.secure.DATA],
)
