"""
Grid Primitives

Basic grid data structures and operations (NOT transformations).
Transformations are learned, not hard-coded.
"""

import numpy as np
from typing import Tuple, List, Set, Optional, Callable
from dataclasses import dataclass


@dataclass
class Cell:
    """Represents a single cell in a grid."""
    row: int
    col: int
    value: int

    def __hash__(self):
        return hash((self.row, self.col, self.value))

    def __eq__(self, other):
        return (self.row, self.col, self.value) == (other.row, other.col, other.value)


class Grid:
    """
    Grid representation with query and manipulation primitives.

    This class provides ONLY basic operations, NOT transformations.
    Transformations are learned by the model.
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize grid from numpy array.

        Args:
            data: 2D numpy array of integers
        """
        self.data = np.array(data, dtype=np.int32)
        self.height, self.width = self.data.shape

    def copy(self) -> 'Grid':
        """Create deep copy of grid."""
        return Grid(self.data.copy())

    # =============================================================================
    # Basic Accessors
    # =============================================================================

    def get_cell(self, row: int, col: int) -> int:
        """Get value at cell (row, col)."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return int(self.data[row, col])
        return None

    def set_cell(self, row: int, col: int, value: int) -> 'Grid':
        """
        Set value at cell (row, col).

        Returns new Grid (immutable operation).
        """
        new_grid = self.copy()
        if 0 <= row < new_grid.height and 0 <= col < new_grid.width:
            new_grid.data[row, col] = value
        return new_grid

    def get_shape(self) -> Tuple[int, int]:
        """Get grid dimensions (height, width)."""
        return (self.height, self.width)

    def get_size(self) -> int:
        """Get total number of cells."""
        return self.height * self.width

    # =============================================================================
    # Cell Queries
    # =============================================================================

    def find_cells_with_value(self, value: int) -> List[Cell]:
        """Find all cells with specific value."""
        cells = []
        for row in range(self.height):
            for col in range(self.width):
                if self.data[row, col] == value:
                    cells.append(Cell(row, col, value))
        return cells

    def count_value(self, value: int) -> int:
        """Count cells with specific value."""
        return int(np.sum(self.data == value))

    def get_unique_values(self) -> Set[int]:
        """Get set of all unique values in grid."""
        return set(int(v) for v in np.unique(self.data))

    def get_value_counts(self) -> dict:
        """Get count of each value."""
        unique, counts = np.unique(self.data, return_counts=True)
        return {int(val): int(count) for val, count in zip(unique, counts)}

    # =============================================================================
    # Spatial Queries
    # =============================================================================

    def get_row(self, row: int) -> Optional[np.ndarray]:
        """Get entire row."""
        if 0 <= row < self.height:
            return self.data[row, :].copy()
        return None

    def get_column(self, col: int) -> Optional[np.ndarray]:
        """Get entire column."""
        if 0 <= col < self.width:
            return self.data[:, col].copy()
        return None

    def get_subgrid(self, row_start: int, row_end: int,
                    col_start: int, col_end: int) -> Optional['Grid']:
        """
        Extract subgrid.

        Args:
            row_start, row_end: Row range (inclusive start, exclusive end)
            col_start, col_end: Column range (inclusive start, exclusive end)

        Returns:
            New Grid with subgrid data
        """
        if (0 <= row_start < row_end <= self.height and
            0 <= col_start < col_end <= self.width):
            subgrid_data = self.data[row_start:row_end, col_start:col_end].copy()
            return Grid(subgrid_data)
        return None

    def get_neighbors(self, row: int, col: int, include_diagonals: bool = False) -> List[Cell]:
        """
        Get neighboring cells.

        Args:
            row, col: Cell coordinates
            include_diagonals: Include diagonal neighbors

        Returns:
            List of Cell objects for valid neighbors
        """
        neighbors = []

        # Orthogonal neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append(Cell(r, c, int(self.data[r, c])))

        # Diagonal neighbors
        if include_diagonals:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                r, c = row + dr, col + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    neighbors.append(Cell(r, c, int(self.data[r, c])))

        return neighbors

    # =============================================================================
    # Pattern Detection Helpers (NOT transformations, just queries)
    # =============================================================================

    def is_uniform(self) -> bool:
        """Check if all cells have same value."""
        return len(self.get_unique_values()) == 1

    def get_background_color(self) -> int:
        """Get most common value (likely background)."""
        counts = self.get_value_counts()
        return max(counts.items(), key=lambda x: x[1])[0]

    def get_foreground_cells(self, background: Optional[int] = None) -> List[Cell]:
        """
        Get all non-background cells.

        Args:
            background: Background value (if None, uses most common)

        Returns:
            List of Cell objects
        """
        if background is None:
            background = self.get_background_color()

        cells = []
        for row in range(self.height):
            for col in range(self.width):
                value = self.data[row, col]
                if value != background:
                    cells.append(Cell(row, col, int(value)))
        return cells

    def get_bounding_box(self, cells: List[Cell]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of cells.

        Args:
            cells: List of Cell objects

        Returns:
            (min_row, max_row, min_col, max_col) or None if no cells
        """
        if not cells:
            return None

        min_row = min(c.row for c in cells)
        max_row = max(c.row for c in cells)
        min_col = min(c.col for c in cells)
        max_col = max(c.col for c in cells)

        return (min_row, max_row, min_col, max_col)

    # =============================================================================
    # Grid Comparison
    # =============================================================================

    def equals(self, other: 'Grid') -> bool:
        """Check if two grids are identical."""
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def difference_mask(self, other: 'Grid') -> Optional[np.ndarray]:
        """
        Get boolean mask of differences.

        Args:
            other: Grid to compare with

        Returns:
            Boolean array where True indicates difference, or None if shapes differ
        """
        if self.get_shape() != other.get_shape():
            return None
        return self.data != other.data

    def count_differences(self, other: 'Grid') -> Optional[int]:
        """
        Count number of cells that differ.

        Returns:
            Number of different cells, or None if shapes differ
        """
        mask = self.difference_mask(other)
        if mask is None:
            return None
        return int(np.sum(mask))

    # =============================================================================
    # Utilities
    # =============================================================================

    def to_numpy(self) -> np.ndarray:
        """Get underlying numpy array."""
        return self.data.copy()

    def __repr__(self):
        return f"Grid(shape={self.get_shape()}, values={sorted(self.get_unique_values())})"

    def __str__(self):
        """Pretty print grid."""
        lines = []
        for row in self.data:
            lines.append(" ".join(str(cell) for cell in row))
        return "\n".join(lines)

    def __eq__(self, other):
        return self.equals(other)

    def __hash__(self):
        # Hash based on grid content (for use in sets/dicts)
        return hash(self.data.tobytes())


# =============================================================================
# Grid Factory Functions
# =============================================================================

def create_empty_grid(height: int, width: int, fill_value: int = 0) -> Grid:
    """
    Create empty grid filled with value.

    Args:
        height: Grid height
        width: Grid width
        fill_value: Value to fill with

    Returns:
        New Grid
    """
    data = np.full((height, width), fill_value, dtype=np.int32)
    return Grid(data)


def create_grid_like(template: Grid, fill_value: int = 0) -> Grid:
    """
    Create grid with same shape as template.

    Args:
        template: Grid to match shape
        fill_value: Value to fill with

    Returns:
        New Grid with same shape
    """
    height, width = template.get_shape()
    return create_empty_grid(height, width, fill_value)


def grids_from_numpy(arrays: List[np.ndarray]) -> List[Grid]:
    """Convert list of numpy arrays to Grids."""
    return [Grid(arr) for arr in arrays]
