from kube_jobs import storage, submit_job


# No defaults in configs/explainability/tile_r2_check_cls.yaml on purpose -
# one job per finished nmf_fit.py (kind=cls) run to check, same convention as
# scripts/explainability/tile_r2_check.py. h_mlflow_uri must be that exact
# run's h.parquet (grade/n_components/scale_power all three determine w_dir
# too - see nmf_fit.yaml's output_dir convention).
grade = ...
n_components = ...
scale_power = ...
h_mlflow_uri = ...

submit_job(
    job_name=f"ulcerative-colitis-tile-r2-check-cls-grade{grade}-k{n_components}-sp{scale_power}-...",
    username=...,
    public=False,
    # Modest on purpose, same reasoning as scripts/explainability/
    # tile_r2_ols_check.py: no patch-token stream - only nmf_fit.py's already
    # -saved local w.f32.npy/w_metadata.parquet, an existing kind=patch
    # tile_r2_check.py run's tile_features.parquet (just its `m` column),
    # one grade's small CLS token partition, and plain numpy/pandas linear
    # algebra (OLS via np.linalg.lstsq).
    cpu=2,
    memory="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.tile_r2_check_cls "
        f"grade={grade} n_components={n_components} scale_power={scale_power} "
        f"h.mlflow_uri={h_mlflow_uri}",
    ],
    storage=[storage.secure.PROJECTS],
)
