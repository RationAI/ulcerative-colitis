from kube_jobs import storage, submit_job


# No default n_components in configs/explainability/nmf_fit.yaml on purpose -
# the plan is to sweep several K (roughly 4-12) and pick/refine with the
# pathologist, not commit to one upfront. Edit this per submission (one job
# per K), same as `username` below.
n_components = ...

# No default scale_power either - 1.0 (IQR, original), 0.5 (sqrt(IQR),
# gentler), 0.0 (no scaling) are the three being compared, per
# explainability-status memory's finding that |Theta_m|*IQR (each patch
# dimension's actual contribution to the trained classifiers' logits) is
# *positively* correlated with IQR, i.e. plain IQR scaling may be
# suppressing exactly the dimensions that matter most. One job per
# (n_components, scale_power) pair - edit both per submission.
scale_power = ...

submit_job(
    job_name=f"ulcerative-colitis-nmf-fit-k{n_components}-sp{scale_power}-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/nmf_fit.py's ray.init(). Same root cause as
    # scripts/explainability/patch_statistics.py's identical comment
    # (oversized parquet row groups, not this script's own logic) - this job
    # was OOM-killed once already before num_cpus was added.
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.nmf_fit "
        f"n_components={n_components} nmf.scale_power={scale_power}",
    ],
    storage=[storage.secure.PROJECTS],
)
