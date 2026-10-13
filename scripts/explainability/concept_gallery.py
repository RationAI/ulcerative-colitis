from kube_jobs import storage, submit_job


# Experiment under configs/experiment/explainability/concept_gallery/ -
# "<dictionary>/<evaluation>", e.g. "tree_min_gain_0.005/test_preliminary".
# Run the matching concept_completeness experiment first (Q1 panel).
experiment = ...

submit_job(
    job_name=f"uc-gallery-{experiment.replace('/', '-').replace('_', '-')}-...",
    username=...,
    public=False,
    # Streams the full embeddings corpus once, then reads a few hundred tiles
    # from the WSIs - keep cpu in sync with num_cpus in
    # explainability/concept_gallery.py's ray.init().
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run --active python -m explainability.concept_gallery "
        f"+experiment=explainability/concept_gallery/{experiment}",
    ],
    # DATA too: the example tiles are read from the WSIs under /mnt/data.
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
