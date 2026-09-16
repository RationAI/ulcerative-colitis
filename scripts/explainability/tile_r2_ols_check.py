from kube_jobs import storage, submit_job


# No defaults in configs/explainability/tile_r2_ols_check.yaml on purpose -
# one job per finished tile_r2_check.py run to extend, same convention as
# scripts/explainability/tile_r2_check.py. h.mlflow_uri must be the exact
# same h that tile_r2_check run was itself checked against.
grade = ...
n_components = ...
h_mlflow_uri = ...

submit_job(
    job_name=f"ulcerative-colitis-tile-r2-ols-check-grade{grade}-k{n_components}-...",
    username=...,
    public=False,
    # Modest on purpose: no ray, no patch-token reads - only a local
    # tile_features.parquet (already pooled by the tile_r2_check run being
    # extended), one grade's small CLS token partition, and plain
    # numpy/pandas linear algebra (OLS via np.linalg.lstsq).
    cpu=2,
    memory="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.tile_r2_ols_check "
        f"grade={grade} n_components={n_components} h.mlflow_uri={h_mlflow_uri}",
    ],
    storage=[storage.secure.PROJECTS],
)
