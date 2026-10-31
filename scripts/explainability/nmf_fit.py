from kube_jobs import storage, submit_job


# Experiment under configs/experiment/explainability/nmf_fit/ - e.g. "semi_nmf_k16"
# (method, n_components, nmf.scale_power). One job per experiment.
experiment = ...
# Hydra overrides applied on top of the experiment, space-separated, e.g.
# "nmf.random_state=1"; "" for none.
# Put a short tag for them in the job name below.
overrides = ...

submit_job(
    job_name=f"uc-nmf-fit-{experiment.replace('/', '-').replace('_', '-')}-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/nmf_fit.py's ray.init() (conservative carry-over from
    # the oversized-row-group OOM on the old embeddings_xai patch tables).
    cpu=8,
    memory="48Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.nmf_fit "
        f"+experiment=explainability/nmf_fit/{experiment} {overrides}",
    ],
    storage=[storage.secure.PROJECTS],
)
