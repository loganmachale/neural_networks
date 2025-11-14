"""
ARC Data Loader

Loads and parses ARC challenge problems from JSON files.
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np


class ARCProblem:
    """Represents a single ARC problem."""

    def __init__(self, problem_id: str, train_pairs: List[Dict], test_pairs: List[Dict]):
        """
        Initialize ARC problem.

        Args:
            problem_id: Unique identifier (filename without .json)
            train_pairs: List of {"input": grid, "output": grid} dicts
            test_pairs: List of {"input": grid, "output": grid} dicts (output may be None)
        """
        self.problem_id = problem_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs

    def get_train_examples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get training examples as numpy arrays.

        Returns:
            List of (input_grid, output_grid) tuples
        """
        examples = []
        for pair in self.train_pairs:
            input_grid = np.array(pair['input'], dtype=np.int32)
            output_grid = np.array(pair['output'], dtype=np.int32)
            examples.append((input_grid, output_grid))
        return examples

    def get_test_inputs(self) -> List[np.ndarray]:
        """
        Get test inputs as numpy arrays.

        Returns:
            List of input grids
        """
        return [np.array(pair['input'], dtype=np.int32) for pair in self.test_pairs]

    def get_test_outputs(self) -> List[np.ndarray]:
        """
        Get test outputs as numpy arrays (if available).

        Returns:
            List of output grids (may contain None for evaluation set)
        """
        outputs = []
        for pair in self.test_pairs:
            if 'output' in pair and pair['output'] is not None:
                outputs.append(np.array(pair['output'], dtype=np.int32))
            else:
                outputs.append(None)
        return outputs

    def __repr__(self):
        return (f"ARCProblem(id={self.problem_id}, "
                f"train={len(self.train_pairs)}, "
                f"test={len(self.test_pairs)})")


class ARCDataLoader:
    """Loads ARC problems from data directory."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            data_dir: Path to data directory containing training/ and evaluation/
        """
        self.data_dir = data_dir
        self.training_dir = os.path.join(data_dir, "training")
        self.evaluation_dir = os.path.join(data_dir, "evaluation")

    def load_problem(self, problem_id: str, split: str = "training") -> ARCProblem:
        """
        Load a single problem by ID.

        Args:
            problem_id: Problem ID (with or without .json extension)
            split: "training" or "evaluation"

        Returns:
            ARCProblem object
        """
        if not problem_id.endswith('.json'):
            problem_id = f"{problem_id}.json"

        if split == "training":
            filepath = os.path.join(self.training_dir, problem_id)
        else:
            filepath = os.path.join(self.evaluation_dir, problem_id)

        with open(filepath, 'r') as f:
            data = json.load(f)

        problem_name = problem_id.replace('.json', '')
        return ARCProblem(
            problem_id=problem_name,
            train_pairs=data.get('train', []),
            test_pairs=data.get('test', [])
        )

    def load_all_problems(self, split: str = "training", limit: int = None) -> List[ARCProblem]:
        """
        Load all problems from a split.

        Args:
            split: "training" or "evaluation"
            limit: Maximum number of problems to load (None for all)

        Returns:
            List of ARCProblem objects
        """
        if split == "training":
            problem_dir = self.training_dir
        else:
            problem_dir = self.evaluation_dir

        problem_files = sorted([f for f in os.listdir(problem_dir) if f.endswith('.json')])

        if limit is not None:
            problem_files = problem_files[:limit]

        problems = []
        for filename in problem_files:
            try:
                problem = self.load_problem(filename, split=split)
                problems.append(problem)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        return problems

    def get_problem_stats(self, problem: ARCProblem) -> Dict:
        """
        Get statistics about a problem.

        Args:
            problem: ARCProblem object

        Returns:
            Dictionary with statistics
        """
        train_examples = problem.get_train_examples()

        input_shapes = [inp.shape for inp, _ in train_examples]
        output_shapes = [out.shape for _, out in train_examples]

        all_colors = set()
        for inp, out in train_examples:
            all_colors.update(inp.flatten())
            all_colors.update(out.flatten())

        return {
            'problem_id': problem.problem_id,
            'num_train': len(train_examples),
            'num_test': len(problem.test_pairs),
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'num_colors': len(all_colors),
            'colors_used': sorted(list(all_colors)),
            'min_input_size': min(inp.size for inp, _ in train_examples),
            'max_input_size': max(inp.size for inp, _ in train_examples),
            'min_output_size': min(out.size for _, out in train_examples),
            'max_output_size': max(out.size for _, out in train_examples),
        }


def visualize_grid(grid: np.ndarray, title: str = "Grid"):
    """
    Print a simple text visualization of a grid.

    Args:
        grid: 2D numpy array
        title: Title to print
    """
    print(f"\n{title}:")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print(" ".join(str(cell) for cell in row))
    print("-" * (grid.shape[1] * 2 + 1))


def visualize_problem(problem: ARCProblem, max_examples: int = 3):
    """
    Visualize a problem's examples.

    Args:
        problem: ARCProblem object
        max_examples: Maximum number of examples to show
    """
    print(f"\n{'='*70}")
    print(f"Problem: {problem.problem_id}")
    print(f"{'='*70}")

    train_examples = problem.get_train_examples()

    for i, (inp, out) in enumerate(train_examples[:max_examples]):
        print(f"\nTraining Example {i+1}:")
        visualize_grid(inp, f"Input {inp.shape}")
        visualize_grid(out, f"Output {out.shape}")

    if len(train_examples) > max_examples:
        print(f"\n... and {len(train_examples) - max_examples} more training examples")

    test_inputs = problem.get_test_inputs()
    print(f"\nTest Cases: {len(test_inputs)}")
    for i, test_inp in enumerate(test_inputs[:1]):
        visualize_grid(test_inp, f"Test Input {i+1} {test_inp.shape}")


if __name__ == "__main__":
    # Example usage
    loader = ARCDataLoader()

    print("Loading first problem from training set...")
    problem = loader.load_problem("00576224", split="training")

    print(f"\nProblem loaded: {problem}")

    # Show statistics
    stats = loader.get_problem_stats(problem)
    print(f"\nProblem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Visualize
    visualize_problem(problem)

    # Load multiple problems
    print("\n" + "="*70)
    print("Loading first 5 problems...")
    problems = loader.load_all_problems(split="training", limit=5)
    print(f"Loaded {len(problems)} problems")

    for p in problems:
        stats = loader.get_problem_stats(p)
        print(f"\n{p.problem_id}: {stats['num_train']} train, {stats['num_test']} test, "
              f"{stats['num_colors']} colors, shapes: {stats['input_shapes']}")
