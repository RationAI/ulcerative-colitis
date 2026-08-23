from kube_jobs import storage, submit_job


# No default n_components in configs/explainability/nmf_fit.yaml on purpose -
# the plan is to sweep several K (roughly 4-12) and pick/refine with the
# pathologist, not commit to one upfront. Edit this per submission (one job
# per K), same as `username` below.
n_components = ...

submit_job(
    job_name=f"ulcerative-colitis-nmf-fit-k{n_components}-...",
    username=...,
    public=False,
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.nmf_fit n_components={n_components}",
    ],
    storage=[storage.secure.PROJECTS],
)
