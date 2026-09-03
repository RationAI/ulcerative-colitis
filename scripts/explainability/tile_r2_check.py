from kube_jobs import storage, submit_job


# No defaults in configs/explainability/tile_r2_check.yaml on purpose - one
# job per (grade, n_components) nmf_fit.py run to check, same convention as
# scripts/explainability/nmf_fit.py. h.mlflow_uri must also be set (either
# edited into the yaml or passed as a CLI override below) to that run's own
# h.parquet artifact.
grade = ...
n_components = ...
h_mlflow_uri = ...

submit_job(
    job_name=f"ulcerative-colitis-tile-r2-check-grade{grade}-k{n_components}-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/tile_r2_check.py's ray.init(). Same conservative
    # carry-over as scripts/explainability/nmf_fit.py's identical comment -
    # reads grade_split.py's much smaller-file output, not the original
    # oversized-row-group tables this limit was first measured against.
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.tile_r2_check "
        f"grade={grade} n_components={n_components} h.mlflow_uri={h_mlflow_uri}",
    ],
    storage=[storage.secure.PROJECTS],
)
