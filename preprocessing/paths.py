from pathlib import Path


BASE_FOLDER = Path("/mnt/data/Projects/inflammatory_bowel_dissease/ulcerative_colitis/")
SLIDES_PATH_IKEM = BASE_FOLDER / "data_tiff" / "20x"
DATAFRAME_PATH_IKEM = BASE_FOLDER / "data_czi" / "IBD_AI.csv"
SLIDES_PATH_FTN = BASE_FOLDER / "data_tiff_ftn"
DATAFRAME_PATH_FTN = BASE_FOLDER / "data_czi_ftn" / "IBD_AI_FTN.xlsx"
SLIDES_PATH_KNL_PATOS = BASE_FOLDER / "data_tiff_knl_patos"
DATAFRAME_PATH_KNL_PATOS = BASE_FOLDER / "data_czi_knl_patos" / "IBD_AI_Liberec.xlsx"
TISSUE_MASKS_PATH = BASE_FOLDER / "tissue_masks" / "20x"
QC_MASKS_PATH = BASE_FOLDER / "qc_masks"
EMBEDDINGS_PATH = BASE_FOLDER / "embeddings"
EMBEDDING_REGIONS_PATH = BASE_FOLDER / "embedding_regions"
