from kube_jobs import storage, submit_job


# Single-job template, same convention as scripts/explainability/nmf_fit.py -
# edit these per submission rather than sweeping automatically.
#
# Unlike h_mlflow_uri/shift_mlflow_uri in an earlier version of this script,
# h.mlflow_uri isn't passed here at all: configs/explainability/
# concept_masks.yaml's `h` is now keyed by n_components (one entry per K
# that's actually been fit) and shift.mlflow_uri already has a real default
# there too - both need to already be filled in before submitting, not
# passed per-run.
n_components = ...  # must have a configs/explainability/concept_masks.yaml h.<n_components> entry

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
        f"uv run --active python -m explainability.concept_masks n_components={n_components}",
    ],
    storage=[storage.secure.PROJECTS],
)
