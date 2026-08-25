from kube_jobs import storage, submit_job


# Single-job template, same convention as scripts/explainability/nmf_fit.py -
# edit these per submission rather than sweeping automatically.
split = ...  # e.g. "test_preliminary" - the overlapping-tile split this script is for
n_components = ...  # must match h_mlflow_uri's actual K
h_mlflow_uri = ...  # a specific nmf_fit run's h.parquet artifact URI
shift_mlflow_uri = ...  # the patch_statistics run h_mlflow_uri was itself shifted with

submit_job(
    job_name=f"ulcerative-colitis-concept-masks-k{n_components}-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/concept_masks.py's ray.init(). Same root cause as
    # scripts/explainability/{patch_statistics,nmf_fit}.py's identical
    # comment (oversized parquet row groups, not this script's own logic).
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active python -m explainability.concept_masks "
        f"split={split} n_components={n_components} "
        f"h.mlflow_uri={h_mlflow_uri} shift.mlflow_uri={shift_mlflow_uri}",
    ],
    storage=[storage.secure.PROJECTS],
)
