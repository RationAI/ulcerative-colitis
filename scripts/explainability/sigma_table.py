from kube_jobs import storage, submit_job


# Unlike nmf_fit.py/concept_masks.py's launchers, no per-run n_components/
# scale_power to edit here - explainability/sigma_table.py sweeps the whole
# configs/explainability/sigma_table.yaml h dict (all K x scale_power combos
# with a finished nmf_fit run) and all three checkpoints in one job.
submit_job(
    job_name="ulcerative-colitis-sigma-table-...",
    username=...,
    public=False,
    # Modest on purpose: no ray, no patch-token parquet reads (the oversized-
    # row-group OOM that forces cpu=8 in patch_statistics.py/nmf_fit.py/
    # concept_masks.py doesn't apply here) - this job only downloads small
    # h.parquet files and three lightning checkpoints from mlflow and does
    # plain numpy/pandas linear algebra.
    cpu=2,
    memory="8Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active python -m explainability.sigma_table",
    ],
    storage=[storage.secure.PROJECTS],
)
