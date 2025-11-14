"""
Logical Operators - Hard-Coded Primitives

These are the fundamental logical operations that allow composition
of learned transformations. They are HARD-CODED as language primitives.

Examples:
- "transform1 AND transform2"
- "IF cell(0,0) == 5 THEN transform1 ELSE transform2"
- "IF count(color=3) > 5 THEN transform1"
"""

from typing import Callable, Any, List, Optional, Union
from abc import ABC, abstractmethod
from grid_primitives import Grid, Cell


# =============================================================================
# Predicate Base Class
# =============================================================================

class Predicate(ABC):
    """Base class for predicates (boolean-valued functions)."""

    @abstractmethod
    def evaluate(self, grid: Grid, **context) -> bool:
        """
        Evaluate predicate on grid.

        Args:
            grid: Grid to evaluate on
            **context: Additional context (e.g., cell positions)

        Returns:
            Boolean result
        """
        pass

    def __and__(self, other: 'Predicate') -> 'Predicate':
        """Allow 'pred1 & pred2' syntax."""
        return AndPredicate(self, other)

    def __or__(self, other: 'Predicate') -> 'Predicate':
        """Allow 'pred1 | pred2' syntax."""
        return OrPredicate(self, other)

    def __invert__(self) -> 'Predicate':
        """Allow '~pred' syntax for NOT."""
        return NotPredicate(self)


# =============================================================================
# Logical Combination Operators (AND, OR, NOT)
# =============================================================================

class AndPredicate(Predicate):
    """Logical AND of two predicates."""

    def __init__(self, left: Predicate, right: Predicate):
        self.left = left
        self.right = right

    def evaluate(self, grid: Grid, **context) -> bool:
        return self.left.evaluate(grid, **context) and self.right.evaluate(grid, **context)

    def __repr__(self):
        return f"({self.left} AND {self.right})"


class OrPredicate(Predicate):
    """Logical OR of two predicates."""

    def __init__(self, left: Predicate, right: Predicate):
        self.left = left
        self.right = right

    def evaluate(self, grid: Grid, **context) -> bool:
        return self.left.evaluate(grid, **context) or self.right.evaluate(grid, **context)

    def __repr__(self):
        return f"({self.left} OR {self.right})"


class NotPredicate(Predicate):
    """Logical NOT of a predicate."""

    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def evaluate(self, grid: Grid, **context) -> bool:
        return not self.predicate.evaluate(grid, **context)

    def __repr__(self):
        return f"(NOT {self.predicate})"


# =============================================================================
# Cell Value Predicates
# =============================================================================

class CellEquals(Predicate):
    """Check if cell at (row, col) equals value."""

    def __init__(self, row: int, col: int, value: int):
        self.row = row
        self.col = col
        self.value = value

    def evaluate(self, grid: Grid, **context) -> bool:
        cell_value = grid.get_cell(self.row, self.col)
        if cell_value is None:
            return False
        return cell_value == self.value

    def __repr__(self):
        return f"cell({self.row},{self.col}) == {self.value}"


class CellNotEquals(Predicate):
    """Check if cell at (row, col) not equals value."""

    def __init__(self, row: int, col: int, value: int):
        self.row = row
        self.col = col
        self.value = value

    def evaluate(self, grid: Grid, **context) -> bool:
        cell_value = grid.get_cell(self.row, self.col)
        if cell_value is None:
            return False
        return cell_value != self.value

    def __repr__(self):
        return f"cell({self.row},{self.col}) != {self.value}"


class AnyCellEquals(Predicate):
    """Check if any cell equals value."""

    def __init__(self, value: int):
        self.value = value

    def evaluate(self, grid: Grid, **context) -> bool:
        return self.value in grid.get_unique_values()

    def __repr__(self):
        return f"any_cell == {self.value}"


# =============================================================================
# Count Predicates
# =============================================================================

class CountEquals(Predicate):
    """Check if count of value equals target."""

    def __init__(self, value: int, target: int):
        self.value = value
        self.target = target

    def evaluate(self, grid: Grid, **context) -> bool:
        return grid.count_value(self.value) == self.target

    def __repr__(self):
        return f"count({self.value}) == {self.target}"


class CountGreaterThan(Predicate):
    """Check if count of value > threshold."""

    def __init__(self, value: int, threshold: int):
        self.value = value
        self.threshold = threshold

    def evaluate(self, grid: Grid, **context) -> bool:
        return grid.count_value(self.value) > self.threshold

    def __repr__(self):
        return f"count({self.value}) > {self.threshold}"


class CountLessThan(Predicate):
    """Check if count of value < threshold."""

    def __init__(self, value: int, threshold: int):
        self.value = value
        self.threshold = threshold

    def evaluate(self, grid: Grid, **context) -> bool:
        return grid.count_value(self.value) < self.threshold

    def __repr__(self):
        return f"count({self.value}) < {self.threshold}"


# =============================================================================
# Grid Property Predicates
# =============================================================================

class GridShapeEquals(Predicate):
    """Check if grid shape equals (height, width)."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def evaluate(self, grid: Grid, **context) -> bool:
        return grid.get_shape() == (self.height, self.width)

    def __repr__(self):
        return f"shape == ({self.height},{self.width})"


class GridIsSquare(Predicate):
    """Check if grid is square."""

    def evaluate(self, grid: Grid, **context) -> bool:
        h, w = grid.get_shape()
        return h == w

    def __repr__(self):
        return "is_square"


class GridIsUniform(Predicate):
    """Check if all cells have same value."""

    def evaluate(self, grid: Grid, **context) -> bool:
        return grid.is_uniform()

    def __repr__(self):
        return "is_uniform"


# =============================================================================
# Transform Base Class
# =============================================================================

class Transform(ABC):
    """
    Base class for grid transformations.

    Transformations are LEARNED, not hard-coded.
    This is just the interface.
    """

    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """
        Apply transformation to grid.

        Args:
            grid: Input grid

        Returns:
            Transformed grid
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass


# =============================================================================
# Conditional Transform (IF-THEN-ELSE)
# =============================================================================

class ConditionalTransform(Transform):
    """
    IF-THEN-ELSE conditional transformation.

    Example:
        IF cell(0,0) == 5 THEN transform_a ELSE transform_b
    """

    def __init__(self, condition: Predicate,
                 then_transform: Transform,
                 else_transform: Optional[Transform] = None):
        """
        Initialize conditional transform.

        Args:
            condition: Predicate to evaluate
            then_transform: Transform to apply if condition is True
            else_transform: Transform to apply if condition is False (optional)
        """
        self.condition = condition
        self.then_transform = then_transform
        self.else_transform = else_transform

    def apply(self, grid: Grid) -> Grid:
        if self.condition.evaluate(grid):
            return self.then_transform.apply(grid)
        elif self.else_transform is not None:
            return self.else_transform.apply(grid)
        else:
            # If no else branch, return grid unchanged
            return grid

    def __repr__(self):
        if self.else_transform:
            return f"IF {self.condition} THEN {self.then_transform} ELSE {self.else_transform}"
        else:
            return f"IF {self.condition} THEN {self.then_transform}"


# =============================================================================
# Composite Transform (AND - sequential application)
# =============================================================================

class CompositeTransform(Transform):
    """
    Composite transformation (sequential application).

    Example:
        transform1 AND transform2  (apply transform1, then transform2)
    """

    def __init__(self, transforms: List[Transform]):
        """
        Initialize composite transform.

        Args:
            transforms: List of transforms to apply in sequence
        """
        self.transforms = transforms

    def apply(self, grid: Grid) -> Grid:
        current = grid
        for transform in self.transforms:
            current = transform.apply(current)
        return current

    def __repr__(self):
        return " AND ".join(str(t) for t in self.transforms)


# =============================================================================
# Identity Transform (no-op)
# =============================================================================

class IdentityTransform(Transform):
    """Identity transformation (returns grid unchanged)."""

    def apply(self, grid: Grid) -> Grid:
        return grid

    def __repr__(self):
        return "IDENTITY"


# =============================================================================
# Helper Functions for Building Logical Expressions
# =============================================================================

def cell_eq(row: int, col: int, value: int) -> Predicate:
    """Shorthand for CellEquals."""
    return CellEquals(row, col, value)


def cell_neq(row: int, col: int, value: int) -> Predicate:
    """Shorthand for CellNotEquals."""
    return CellNotEquals(row, col, value)


def any_cell_eq(value: int) -> Predicate:
    """Shorthand for AnyCellEquals."""
    return AnyCellEquals(value)


def count_eq(value: int, target: int) -> Predicate:
    """Shorthand for CountEquals."""
    return CountEquals(value, target)


def count_gt(value: int, threshold: int) -> Predicate:
    """Shorthand for CountGreaterThan."""
    return CountGreaterThan(value, threshold)


def count_lt(value: int, threshold: int) -> Predicate:
    """Shorthand for CountLessThan."""
    return CountLessThan(value, threshold)


def if_then_else(condition: Predicate, then_transform: Transform,
                 else_transform: Optional[Transform] = None) -> Transform:
    """Shorthand for ConditionalTransform."""
    return ConditionalTransform(condition, then_transform, else_transform)


def compose(*transforms: Transform) -> Transform:
    """Shorthand for CompositeTransform."""
    return CompositeTransform(list(transforms))


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from grid_primitives import create_empty_grid

    # Create test grid
    grid = create_empty_grid(3, 3, fill_value=0)
    grid = grid.set_cell(0, 0, 5)
    grid = grid.set_cell(1, 1, 3)

    print("Test Grid:")
    print(grid)
    print()

    # Test predicates
    pred1 = cell_eq(0, 0, 5)
    pred2 = cell_eq(1, 1, 3)
    pred3 = count_gt(0, 5)

    print(f"{pred1} => {pred1.evaluate(grid)}")  # True
    print(f"{pred2} => {pred2.evaluate(grid)}")  # True
    print(f"{pred3} => {pred3.evaluate(grid)}")  # True (7 zeros)
    print()

    # Test logical combinations
    combined = pred1 & pred2
    print(f"{combined} => {combined.evaluate(grid)}")  # True

    combined2 = pred1 | cell_eq(0, 0, 9)
    print(f"{combined2} => {combined2.evaluate(grid)}")  # True

    negated = ~pred1
    print(f"{negated} => {negated.evaluate(grid)}")  # False
    print()

    # Test more complex expressions
    complex_pred = (cell_eq(0, 0, 5) | cell_eq(0, 0, 9)) & count_gt(0, 5)
    print(f"{complex_pred} => {complex_pred.evaluate(grid)}")  # True
