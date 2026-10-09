from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-embedding-statistics-...",
    username=...,
    public=False,
    # Deliberately low - keep in sync with num_cpus in
    # explainability/token_statistics.py's ray.init(). Confirmed necessary on
    # the old embeddings_xai patch tables (oversized parquet row groups, see
    # that file's comment); kept as the conservative default here.
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        # --extra explainability pulls in pytdigest, which is *not* in the
        # base dependencies (see pyproject.toml) specifically so other jobs'
        # plain `uv sync --frozen` don't have to build it. It compiles a C
        # extension from source (no prebuilt wheel for this platform) - the
        # interpreter's baked-in CFLAGS include a clang-only flag
        # (-fdebug-default-version=4) that this GCC-based toolchain rejects.
        # Stripped here the same way it had to be stripped to install it
        # locally; unverified whether this container's toolchain hits the
        # identical issue - if `uv sync` still fails here, this is the first
        # thing to check.
        "CFLAGS='-fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -fPIC' uv sync --frozen --extra explainability",
        "uv run --active python -m explainability.token_statistics",
    ],
    storage=[storage.secure.PROJECTS],
)
