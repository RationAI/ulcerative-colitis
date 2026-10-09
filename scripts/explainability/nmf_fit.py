from kube_jobs import storage, submit_job


# No default n_components in configs/explainability/nmf_fit.yaml on purpose -
# the plan is to sweep several K (roughly 4-12) and pick/refine with the
# pathologist, not commit to one upfront. Edit this per submission (one job
# per K), same as `username` below.
n_components = ...

# No default scale_power either - 1.0 (IQR), 0.5 (sqrt(IQR), gentler), 0.0
# (no scaling). One job per (n_components, scale_power) pair.
scale_power = ...

# token_statistics.py run over the per-tile embeddings - its
# percentile_stats.parquet artifact URI (no default in nmf_fit.yaml).
shift_mlflow_uri = ...

submit_job(
    job_name=f"ulcerative-colitis-nmf-fit-embedding-k{n_components}-sp{scale_power}-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/nmf_fit.py's ray.init() (conservative carry-over from
    # the oversized-row-group OOM on the old embeddings_xai patch tables).
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.nmf_fit "
        f"n_components={n_components} nmf.scale_power={scale_power} "
        f"shift.mlflow_uri={shift_mlflow_uri}",
    ],
    storage=[storage.secure.PROJECTS],
)
