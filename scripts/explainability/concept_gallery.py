from kube_jobs import storage, submit_job


# Output dir of the nmf_fit.py / nmf_tree.py run to show - no default in
# configs/explainability/concept_gallery.yaml, one job per fit.
w_dir = ...

submit_job(
    job_name="ulcerative-colitis-concept-gallery-...",
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
        f"uv run --active python -m explainability.concept_gallery w_dir={w_dir}",
    ],
    # DATA too: the example tiles are read from the WSIs under /mnt/data.
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
