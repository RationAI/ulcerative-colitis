from dataclasses import asdict

from preprocessing.typing import UlcerativeColitisTileMetadata


Grid = list[list[UlcerativeColitisTileMetadata | None]]


def extreme_coordinates(
    tiles: list[UlcerativeColitisTileMetadata],
) -> tuple[int, int, int, int]:
    """Finds the extreme coordinates of the tiles.

    Args:
        tiles: List of tile metadata.

    Returns:
        A tuple containing the minimum x, maximum x, minimum y, and maximum y coordinates.
    """
    min_x = min(tile.x for tile in tiles)
    max_x = max(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    max_y = max(tile.y for tile in tiles)

    return min_x, max_x, min_y, max_y


def create_grid(
    tiles: list[UlcerativeColitisTileMetadata],
    stride: int,
) -> Grid:
    """Creates a grid of tiles based on their spatial arrangement.

    Args:
        tiles: List of tile metadata.
        stride: Stride used for tiling.

    Returns:
        A grid (list of lists) where each element is either a UlcerativeColitisTileMetadata or None.
    """
    min_x, max_x, min_y, max_y = extreme_coordinates(tiles)
    width = (max_x - min_x) // stride + 1
    height = (max_y - min_y) // stride + 1

    grid: Grid = [[None for _ in range(width)] for _ in range(height)]

    for tile_metadata in tiles:
        x_index = (tile_metadata.x - min_x) // stride
        y_index = (tile_metadata.y - min_y) // stride

        assert 0 <= x_index < width and 0 <= y_index < height

        grid[y_index][x_index] = UlcerativeColitisTileMetadata(**asdict(tile_metadata))

    return grid


def dfs(grid: Grid, x: int, y: int, label: int, reach: int) -> None:
    """Depth-first search to label connected components in the grid.

    The function accepts only coordinates of tiles that are within the grid bounds.

    Args:
        grid: The grid of tiles.
        x: The x-coordinate of the current tile.
        y: The y-coordinate of the current tile.
        label: The label to assign to the connected component.
        reach: The maximum L_infinity distance to consider for connectivity.
    """
    stack = [(x, y)]
    tile = grid[y][x]
    assert tile is not None, "Tile should not be None in DFS"
    tile.region = label

    while stack:
        x, y = stack.pop()

        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if dx == 0 and dy == 0:
                    continue

                neigh_x = x + dx
                neigh_y = y + dy

                if 0 <= neigh_x < len(grid[0]) and 0 <= neigh_y < len(grid):
                    neigh_tile = grid[neigh_y][neigh_x]
                    if neigh_tile is not None and neigh_tile.region == -1:
                        neigh_tile.region = label
                        stack.append((neigh_x, neigh_y))


def add_regions(
    tiles: list[UlcerativeColitisTileMetadata],
    tile_size: int,
    stride: int,
) -> list[UlcerativeColitisTileMetadata]:
    """Adds tissue region labels to tiles based on their spatial arrangement.

    Args:
        tiles: List of tile metadata.
        tile_size: Size of the tile.
        stride: Stride used for tiling.

    Returns:
        List of UlcerativeColitisTileMetadata with tissue region labels.
    """
    grid = create_grid(tiles, stride)

    reach = tile_size // stride

    label = 0
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            tile = grid[y][x]
            if tile is not None and tile.region == -1:
                dfs(grid, x, y, label=label, reach=reach)
                label += 1

    return [tile for row in grid for tile in row if tile is not None]
