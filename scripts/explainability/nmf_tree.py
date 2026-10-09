from kube_jobs import storage, submit_job


# "nmf" or "semi_nmf" - no default in configs/explainability/nmf_tree.yaml.
method = ...
# IQR ** scale_power - no default either.
scale_power = ...
# Optional overrides of configs/explainability/nmf_tree.yaml's tree block,
# e.g. "tree.tau_diff=0.05 tree.supervised=false" - empty for the defaults.
tree_overrides = ...

submit_job(
    job_name=f"ulcerative-colitis-nmf-tree-{method.replace('_', '-')}-sp{scale_power}-...",
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
        f"method={method} scale_power={scale_power} "
        f"{tree_overrides}",
    ],
    storage=[storage.secure.PROJECTS],
)
