from pathlib import Path


BASE_FOLDER = Path("/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/")
SLIDES_PATH = BASE_FOLDER / "data_tiff" / "20x"
DATAFRAME_PATH = BASE_FOLDER / "data_czi" / "IBD_AI.csv"
TISSUE_MASKS_PATH = BASE_FOLDER / "tissue_masks" / "20x"
EMBEDDINGS_PATH = BASE_FOLDER / "embeddings"
EMBEDDING_REGIONS_PATH = BASE_FOLDER / "embedding_regions"
