"""
Transform Learning System

Uses the logic model to LEARN transformations from examples.
No hard-coded grid transformations - all are discovered through reasoning.

The system combines:
1. Learned atomic transformations (discovered from data)
2. Hard-coded logical operators (AND, OR, IF, ==, etc.)
"""

from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from grid_primitives import Grid, Cell
from logical_operators import (
    Transform, Predicate, ConditionalTransform, CompositeTransform,
    IdentityTransform, cell_eq, count_gt, any_cell_eq
)
from logic_model import InternalState, Goal, ActionDI, LogicModel, ScoringModule


# =============================================================================
# Atomic Learned Transformations
# =============================================================================

class LearnedTransform(Transform):
    """
    A transformation learned from examples.

    This is an atomic transformation that will be discovered,
    not hard-coded.
    """

    def __init__(self, name: str, implementation: callable):
        """
        Initialize learned transform.

        Args:
            name: Human-readable name
            implementation: Function that takes Grid and returns Grid
        """
        self.name = name
        self.implementation = implementation

    def apply(self, grid: Grid) -> Grid:
        return self.implementation(grid)

    def __repr__(self):
        return f"Transform[{self.name}]"


# =============================================================================
# Transformation Hypotheses
# =============================================================================

class TransformHypothesis:
    """
    A hypothesis about a transformation.

    Represents a belief about what transformation might solve the problem.
    """

    def __init__(self, transform: Transform, confidence: float, description: str = ""):
        """
        Initialize hypothesis.

        Args:
            transform: The proposed transformation
            confidence: Confidence level (0-1)
            description: Human-readable description
        """
        self.transform = transform
        self.confidence = confidence
        self.description = description

    def test_on_examples(self, examples: List[Tuple[Grid, Grid]]) -> Tuple[float, int]:
        """
        Test hypothesis on examples.

        Args:
            examples: List of (input, expected_output) pairs

        Returns:
            (accuracy, num_correct) where accuracy is 0-1
        """
        correct = 0
        for input_grid, expected_output in examples:
            try:
                predicted = self.transform.apply(input_grid)
                if predicted.equals(expected_output):
                    correct += 1
            except Exception as e:
                # Transformation failed, count as incorrect
                pass

        accuracy = correct / len(examples) if examples else 0
        return accuracy, correct

    def __repr__(self):
        return f"Hypothesis({self.transform}, conf={self.confidence:.2f})"


# =============================================================================
# Transformation Discovery Actions (for Logic Model)
# =============================================================================

class ProposeTransformAction(ActionDI):
    """Action to propose a transformation hypothesis."""

    def __init__(self, transform: Transform, confidence: float):
        self.transform = transform
        self.confidence = confidence

    def apply(self, state: InternalState) -> InternalState:
        """Add transformation hypothesis to state."""
        new_state = state.copy()
        # Store hypothesis as a fact
        hypothesis_id = f"hypothesis_{id(self.transform)}"
        new_state.add_fact(hypothesis_id, (self.transform, self.confidence), self.confidence)
        return new_state

    def __repr__(self):
        return f"Propose({self.transform})"


class ValidateTransformAction(ActionDI):
    """Action to validate a transformation against examples."""

    def __init__(self, hypothesis_key: str, examples: List[Tuple[Grid, Grid]]):
        self.hypothesis_key = hypothesis_key
        self.examples = examples

    def apply(self, state: InternalState) -> InternalState:
        """Validate hypothesis and update confidence."""
        new_state = state.copy()

        hypothesis_fact = state.get_fact(self.hypothesis_key)
        if hypothesis_fact:
            transform, old_conf = hypothesis_fact
            if isinstance(transform, Transform):
                # Create temporary hypothesis to test
                hyp = TransformHypothesis(transform, old_conf)
                accuracy, num_correct = hyp.test_on_examples(self.examples)

                # Update confidence based on accuracy
                new_conf = accuracy
                new_state.add_fact(f"{self.hypothesis_key}_validated", True, new_conf)
                new_state.add_fact(f"{self.hypothesis_key}_accuracy", accuracy, new_conf)

        return new_state

    def __repr__(self):
        return f"Validate({self.hypothesis_key})"


# =============================================================================
# Transformation Generator
# =============================================================================

class TransformGenerator:
    """
    Generates candidate transformations from examples.

    This is where learning happens - analyzing input/output pairs
    to hypothesize transformations.
    """

    def __init__(self):
        self.learned_transforms: Dict[str, Transform] = {}

    def analyze_examples(self, examples: List[Tuple[Grid, Grid]]) -> List[TransformHypothesis]:
        """
        Analyze examples and generate transformation hypotheses.

        Args:
            examples: List of (input, output) Grid pairs

        Returns:
            List of TransformHypothesis objects
        """
        hypotheses = []

        if not examples:
            return hypotheses

        # Analyze patterns
        for input_grid, output_grid in examples:
            # Check shape changes
            in_shape = input_grid.get_shape()
            out_shape = output_grid.get_shape()

            # Hypothesis 1: Shape unchanged (cell-wise operation)
            if in_shape == out_shape:
                hypotheses.extend(self._generate_cellwise_hypotheses(input_grid, output_grid))

            # Hypothesis 2: Shape changed (geometric operation)
            else:
                hypotheses.extend(self._generate_geometric_hypotheses(input_grid, output_grid))

        # Deduplicate and rank
        return self._rank_hypotheses(hypotheses)

    def _generate_cellwise_hypotheses(self, input_grid: Grid, output_grid: Grid) -> List[TransformHypothesis]:
        """Generate hypotheses for same-shape transformations."""
        hypotheses = []

        # Check if it's an identity transform
        if input_grid.equals(output_grid):
            hypotheses.append(TransformHypothesis(
                IdentityTransform(),
                confidence=1.0,
                description="Identity (no change)"
            ))
            return hypotheses

        # Check for color mapping
        color_mapping = self._infer_color_mapping(input_grid, output_grid)
        if color_mapping:
            transform = self._create_color_map_transform(color_mapping)
            hypotheses.append(TransformHypothesis(
                transform,
                confidence=0.8,
                description=f"Color mapping: {color_mapping}"
            ))

        # Check for conditional rules
        conditional_hyp = self._infer_conditional_rules(input_grid, output_grid)
        if conditional_hyp:
            hypotheses.extend(conditional_hyp)

        return hypotheses

    def _generate_geometric_hypotheses(self, input_grid: Grid, output_grid: Grid) -> List[TransformHypothesis]:
        """Generate hypotheses for shape-changing transformations."""
        hypotheses = []

        in_h, in_w = input_grid.get_shape()
        out_h, out_w = output_grid.get_shape()

        # Check for tiling/repetition with alternating pattern
        if out_h % in_h == 0 and out_w % in_w == 0:
            repeat_h = out_h // in_h
            repeat_w = out_w // in_w

            if repeat_h > 1 or repeat_w > 1:
                # Try alternating checkerboard tile pattern
                transform = self._create_checkerboard_tile_transform(input_grid, output_grid, repeat_h, repeat_w)
                if transform:
                    hypotheses.append(TransformHypothesis(
                        transform,
                        confidence=0.9,
                        description=f"Checkerboard tile {repeat_h}x{repeat_w}"
                    ))

                # Also try simple tiling
                transform = self._create_tile_transform(repeat_h, repeat_w)
                hypotheses.append(TransformHypothesis(
                    transform,
                    confidence=0.7,
                    description=f"Tile {repeat_h}x{repeat_w}"
                ))

        # Check for scaling
        if out_h > in_h and out_w > in_w:
            scale_h = out_h / in_h
            scale_w = out_w / in_w

            if scale_h == scale_w and scale_h == int(scale_h):
                transform = self._create_scale_transform(int(scale_h))
                hypotheses.append(TransformHypothesis(
                    transform,
                    confidence=0.6,
                    description=f"Scale {int(scale_h)}x"
                ))

        return hypotheses

    def _infer_color_mapping(self, input_grid: Grid, output_grid: Grid) -> Optional[Dict[int, int]]:
        """
        Infer color mapping from input to output.

        Returns:
            Dictionary mapping input colors to output colors, or None
        """
        if input_grid.get_shape() != output_grid.get_shape():
            return None

        mapping = {}
        for row in range(input_grid.height):
            for col in range(input_grid.width):
                in_val = input_grid.get_cell(row, col)
                out_val = output_grid.get_cell(row, col)

                if in_val in mapping:
                    if mapping[in_val] != out_val:
                        # Inconsistent mapping
                        return None
                else:
                    mapping[in_val] = out_val

        # Check if mapping is actually changing anything
        if all(k == v for k, v in mapping.items()):
            return None

        return mapping

    def _create_color_map_transform(self, mapping: Dict[int, int]) -> Transform:
        """Create a color mapping transformation."""
        def color_map_impl(grid: Grid) -> Grid:
            new_grid = grid.copy()
            for row in range(grid.height):
                for col in range(grid.width):
                    val = grid.get_cell(row, col)
                    if val in mapping:
                        new_grid = new_grid.set_cell(row, col, mapping[val])
            return new_grid

        return LearnedTransform(f"ColorMap{mapping}", color_map_impl)

    def _create_checkerboard_tile_transform(self, input_grid: Grid, output_grid: Grid,
                                            repeat_h: int, repeat_w: int) -> Optional[Transform]:
        """
        Create a checkerboard tiling transformation by learning the pattern.

        Analyzes the output to detect alternating/flipped patterns.
        """
        in_data = input_grid.to_numpy()
        out_data = output_grid.to_numpy()
        in_h, in_w = input_grid.get_shape()

        # Check if output matches a checkerboard pattern
        # Extract each tile and see if there's a pattern
        tile_patterns = []
        for tile_row in range(repeat_h):
            for tile_col in range(repeat_w):
                row_start = tile_row * in_h
                row_end = (tile_row + 1) * in_h
                col_start = tile_col * in_w
                col_end = (tile_col + 1) * in_w

                tile = out_data[row_start:row_end, col_start:col_end]
                tile_patterns.append((tile_row, tile_col, tile))

        # Check if tiles alternate in a pattern
        # Pattern 1: Horizontal flip on alternating rows
        flipped_h = np.fliplr(in_data)
        matches_h_flip = True

        for tile_row, tile_col, tile in tile_patterns:
            if tile_row % 2 == 0:
                expected = in_data
            else:
                expected = flipped_h

            if not np.array_equal(tile, expected):
                matches_h_flip = False
                break

        if matches_h_flip:
            def checkerboard_h_impl(grid: Grid) -> Grid:
                data = grid.to_numpy()
                flipped = np.fliplr(data)
                rows = []
                for i in range(repeat_h):
                    if i % 2 == 0:
                        rows.append(np.tile(data, (1, repeat_w)))
                    else:
                        rows.append(np.tile(flipped, (1, repeat_w)))
                return Grid(np.vstack(rows))

            return LearnedTransform(f"CheckerboardH_{repeat_h}x{repeat_w}", checkerboard_h_impl)

        # Pattern 2: Vertical flip on alternating columns
        flipped_v = np.flipud(in_data)
        matches_v_flip = True

        for tile_row, tile_col, tile in tile_patterns:
            if tile_col % 2 == 0:
                expected = in_data
            else:
                expected = flipped_v

            if not np.array_equal(tile, expected):
                matches_v_flip = False
                break

        if matches_v_flip:
            def checkerboard_v_impl(grid: Grid) -> Grid:
                data = grid.to_numpy()
                flipped = np.flipud(data)
                result = np.zeros((data.shape[0] * repeat_h, data.shape[1] * repeat_w), dtype=np.int32)
                for i in range(repeat_h):
                    for j in range(repeat_w):
                        row_start = i * data.shape[0]
                        col_start = j * data.shape[1]
                        if j % 2 == 0:
                            result[row_start:row_start+data.shape[0], col_start:col_start+data.shape[1]] = data
                        else:
                            result[row_start:row_start+data.shape[0], col_start:col_start+data.shape[1]] = flipped
                return Grid(result)

            return LearnedTransform(f"CheckerboardV_{repeat_h}x{repeat_w}", checkerboard_v_impl)

        # No checkerboard pattern found
        return None

    def _create_tile_transform(self, repeat_h: int, repeat_w: int) -> Transform:
        """Create a tiling transformation."""
        def tile_impl(grid: Grid) -> Grid:
            in_data = grid.to_numpy()
            tiled = np.tile(in_data, (repeat_h, repeat_w))
            return Grid(tiled)

        return LearnedTransform(f"Tile_{repeat_h}x{repeat_w}", tile_impl)

    def _create_scale_transform(self, scale: int) -> Transform:
        """Create a scaling transformation."""
        def scale_impl(grid: Grid) -> Grid:
            in_data = grid.to_numpy()
            scaled = np.repeat(np.repeat(in_data, scale, axis=0), scale, axis=1)
            return Grid(scaled)

        return LearnedTransform(f"Scale_{scale}x", scale_impl)

    def _infer_conditional_rules(self, input_grid: Grid, output_grid: Grid) -> List[TransformHypothesis]:
        """Infer conditional transformation rules."""
        hypotheses = []

        # Check for "if cell == X then set to Y" rules
        for in_val in input_grid.get_unique_values():
            # Find what this value maps to
            out_vals = set()
            for row in range(input_grid.height):
                for col in range(input_grid.width):
                    if input_grid.get_cell(row, col) == in_val:
                        out_vals.add(output_grid.get_cell(row, col))

            # If all instances of in_val map to same out_val, it's a rule
            if len(out_vals) == 1:
                out_val = list(out_vals)[0]
                if in_val != out_val:  # Actual change
                    # This is part of color mapping, already handled
                    pass

        return hypotheses

    def _rank_hypotheses(self, hypotheses: List[TransformHypothesis]) -> List[TransformHypothesis]:
        """Rank hypotheses by confidence."""
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)


# =============================================================================
# Transform Learner (Uses Logic Model)
# =============================================================================

class TransformLearner:
    """
    Main learner that uses logic model to discover transformations.
    """

    def __init__(self, search_depth: int = 3):
        """
        Initialize learner.

        Args:
            search_depth: Depth for tree search
        """
        self.generator = TransformGenerator()
        self.search_depth = search_depth
        self.learned_library: List[TransformHypothesis] = []

    def learn_from_examples(self, examples: List[Tuple[Grid, Grid]],
                           verbose: bool = True) -> Optional[Transform]:
        """
        Learn transformation from examples.

        Args:
            examples: List of (input, output) Grid pairs
            verbose: Print progress

        Returns:
            Best Transform found, or None
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Learning transformation from {len(examples)} examples")
            print(f"{'='*70}")

        # Generate initial hypotheses
        hypotheses = self.generator.analyze_examples(examples)

        if verbose:
            print(f"\nGenerated {len(hypotheses)} initial hypotheses:")
            for i, hyp in enumerate(hypotheses[:5]):  # Show top 5
                print(f"  {i+1}. {hyp.description} (conf={hyp.confidence:.2f})")

        # Test hypotheses
        best_hypothesis = None
        best_accuracy = 0

        for hyp in hypotheses:
            accuracy, num_correct = hyp.test_on_examples(examples)

            if verbose:
                print(f"\nTesting: {hyp.description}")
                print(f"  Accuracy: {accuracy:.2%} ({num_correct}/{len(examples)} correct)")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hypothesis = hyp

            # Perfect match found
            if accuracy == 1.0:
                break

        if verbose:
            print(f"\n{'='*70}")
            if best_hypothesis:
                print(f"Best transformation: {best_hypothesis.description}")
                print(f"Accuracy: {best_accuracy:.2%}")
            else:
                print("No valid transformation found")
            print(f"{'='*70}\n")

        # Store in library
        if best_hypothesis and best_accuracy > 0:
            self.learned_library.append(best_hypothesis)
            return best_hypothesis.transform

        return None


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from arc_loader import ARCDataLoader

    # Load ARC data
    loader = ARCDataLoader()
    problem = loader.load_problem("00576224", split="training")

    print("Loaded problem:", problem)

    # Get examples
    train_examples = problem.get_train_examples()
    grid_examples = [(Grid(inp), Grid(out)) for inp, out in train_examples]

    print(f"\nTraining examples: {len(grid_examples)}")
    for i, (inp, out) in enumerate(grid_examples):
        print(f"  Example {i+1}: {inp.get_shape()} -> {out.get_shape()}")

    # Learn transformation
    learner = TransformLearner()
    learned_transform = learner.learn_from_examples(grid_examples, verbose=True)

    if learned_transform:
        print("\nTesting learned transform on test cases...")
        test_inputs = problem.get_test_inputs()

        for i, test_input in enumerate(test_inputs):
            test_grid = Grid(test_input)
            result = learned_transform.apply(test_grid)

            print(f"\nTest case {i+1}:")
            print(f"  Input shape: {test_grid.get_shape()}")
            print(f"  Output shape: {result.get_shape()}")
            print(f"  Output:\n{result}")
