from kube_jobs import storage, submit_job


# Experiment under configs/experiment/explainability/nmf_tree/ - e.g.
# "min_gain_0.005" or "min_gain_0.002". One job per experiment.
experiment = ...
# Hydra overrides applied on top of the experiment, space-separated, e.g.
# "factorization.tol=0.0001 tree.stability_threshold=0.75 tree.random_state=0"; "" for none.
# Put a short tag for them in the job name below.
overrides = ...

submit_job(
    job_name=f"uc-nmf-tree-{experiment.replace('/', '-').replace('_', '-')}-...",
    username=...,
    public=False,
    # Keep cpu in sync with num_cpus in explainability/nmf_tree.py's
    # ray.init(). Memory: the tree is built on an in-memory sample of slides
    # (~4GB at the default 0.3 fraction) plus per-node copies.
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.nmf_tree "
        f"+experiment=explainability/nmf_tree/{experiment} {overrides}",
    ],
    storage=[storage.secure.PROJECTS],
)
