from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-slide-auc-check-cls-...",
    username=...,
    public=False,
    # Same resource profile as scripts/explainability/embedding_importance.py
    # - not the cheap tile_r2_check_cls.py shape, despite the surrogate
    # itself being z_i-only: this still streams the full patch corpus once
    # (mean_pool_patches) to get real m_i for m_bar and the real baseline
    # forward pass. Keep cpu in sync with num_cpus in
    # explainability/slide_auc_check_cls.py's ray.init().
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active python -m explainability.slide_auc_check_cls",
    ],
    storage=[storage.secure.PROJECTS],
)
