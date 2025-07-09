from dataclasses import dataclass

from rationai.tiling.modules.tile_sources.openslide_tile_source import OpenSlideMetadata
from rationai.tiling.typing import TileMetadata


@dataclass
class UlcerativeColitisSlideMetadata(OpenSlideMetadata):
    nancy_index: int
    location: str
    diagnosis: str


@dataclass
class UlcerativeColitisTileMetadata(TileMetadata):
    region: int = -1
