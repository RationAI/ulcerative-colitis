from kube_jobs import storage, submit_job


# No defaults in configs/explainability/concept_surrogate_check.yaml - one job
# per finished nmf_fit.py run, which these two values (plus split) identify.
n_components = ...
scale_power = ...

submit_job(
    job_name=f"ulcerative-colitis-concept-surrogate-check-k{n_components}-sp{scale_power}-...",
    username=...,
    public=False,
    # Streams the full embeddings corpus once - keep cpu in sync with
    # num_cpus in explainability/concept_surrogate_check.py's ray.init().
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.concept_surrogate_check "
        f"n_components={n_components} scale_power={scale_power}",
    ],
    storage=[storage.secure.PROJECTS],
)
