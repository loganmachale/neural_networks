"""
Demo: ARC Challenge - Transform Learning

Demonstrates the system learning transformations from ARC examples.
"""

from arc_loader import ARCDataLoader, visualize_problem
from grid_primitives import Grid
from transform_learning import TransformLearner
from logical_operators import cell_eq, if_then_else, IdentityTransform


def demo_basic_learning():
    """Demo: Learn transformation from ARC problem."""
    print("\n" + "="*70)
    print("DEMO: Learning Transformation from ARC Examples")
    print("="*70)

    # Load ARC problem
    loader = ARCDataLoader()
    problem = loader.load_problem("00576224", split="training")

    # Visualize the problem
    visualize_problem(problem, max_examples=2)

    # Convert to Grid objects
    train_examples = problem.get_train_examples()
    grid_examples = [(Grid(inp), Grid(out)) for inp, out in train_examples]

    # Learn transformation
    print("\n" + "="*70)
    print("LEARNING PHASE")
    print("="*70)

    learner = TransformLearner()
    learned_transform = learner.learn_from_examples(grid_examples, verbose=True)

    # Test on test cases
    if learned_transform:
        print("\n" + "="*70)
        print("TESTING PHASE")
        print("="*70)

        test_inputs = problem.get_test_inputs()
        test_outputs = problem.get_test_outputs()

        for i, test_input in enumerate(test_inputs):
            test_grid = Grid(test_input)
            predicted = learned_transform.apply(test_grid)

            print(f"\nTest Case {i+1}:")
            print(f"  Input ({test_grid.get_shape()}):")
            print(f"    {test_grid}")
            print(f"\n  Predicted Output ({predicted.get_shape()}):")
            print(f"    {predicted}")

            if test_outputs[i] is not None:
                expected = Grid(test_outputs[i])
                correct = predicted.equals(expected)
                print(f"\n  Correct: {correct}")


def demo_logical_operators():
    """Demo: Hard-coded logical operators with learned transforms."""
    print("\n" + "="*70)
    print("DEMO: Logical Operators (Hard-Coded Primitives)")
    print("="*70)

    from grid_primitives import create_empty_grid
    from transform_learning import LearnedTransform

    # Create test grid
    grid = create_empty_grid(3, 3, fill_value=0)
    grid = grid.set_cell(0, 0, 5)
    grid = grid.set_cell(1, 1, 3)
    grid = grid.set_cell(2, 2, 9)

    print("\nOriginal Grid:")
    print(grid)

    # Define a learned transform (simple example: set all cells to 1)
    def set_all_to_one(g: Grid) -> Grid:
        result = g.copy()
        for r in range(g.height):
            for c in range(g.width):
                result = result.set_cell(r, c, 1)
        return result

    transform_a = LearnedTransform("SetAllToOne", set_all_to_one)

    # Use logical operators to create conditional transform
    # IF cell(0,0) == 5 THEN apply transform_a ELSE do nothing
    conditional = if_then_else(
        condition=cell_eq(0, 0, 5),
        then_transform=transform_a,
        else_transform=IdentityTransform()
    )

    print(f"\nLogical Expression: {conditional}")
    print(f"Evaluating condition: cell(0,0) == 5 => {cell_eq(0, 0, 5).evaluate(grid)}")

    result = conditional.apply(grid)
    print("\nResult after applying conditional transform:")
    print(result)

    # Test with different grid
    grid2 = create_empty_grid(3, 3, fill_value=0)
    grid2 = grid2.set_cell(0, 0, 7)  # Different value at (0,0)

    print("\n" + "-"*70)
    print("Testing with different grid (cell(0,0) = 7):")
    print(grid2)

    print(f"Evaluating condition: cell(0,0) == 5 => {cell_eq(0, 0, 5).evaluate(grid2)}")

    result2 = conditional.apply(grid2)
    print("\nResult (should be unchanged - ELSE branch):")
    print(result2)


def demo_transform_library():
    """Demo: Multiple problems, building transform library."""
    print("\n" + "="*70)
    print("DEMO: Building Transform Library from Multiple Problems")
    print("="*70)

    loader = ARCDataLoader()
    learner = TransformLearner()

    # Learn from first 3 problems
    problem_ids = ["00576224", "007bbfb7", "00d62c1b"]

    for problem_id in problem_ids:
        try:
            problem = loader.load_problem(problem_id, split="training")
            print(f"\n{'='*70}")
            print(f"Problem: {problem_id}")
            print(f"{'='*70}")

            train_examples = problem.get_train_examples()
            grid_examples = [(Grid(inp), Grid(out)) for inp, out in train_examples]

            print(f"Training examples: {len(grid_examples)}")
            for i, (inp, out) in enumerate(grid_examples[:2]):
                print(f"  Example {i+1}: {inp.get_shape()} -> {out.get_shape()}")

            # Learn transformation
            learned = learner.learn_from_examples(grid_examples, verbose=False)

            if learned:
                print(f"[SUCCESS] Learned transformation: {learned}")
            else:
                print(f"[FAILED] Could not learn transformation")

        except Exception as e:
            print(f"Error with {problem_id}: {e}")

    print(f"\n{'='*70}")
    print(f"Transform Library Summary")
    print(f"{'='*70}")
    print(f"Total transforms learned: {len(learner.learned_library)}")
    for i, hyp in enumerate(learner.learned_library):
        print(f"  {i+1}. {hyp.description} (accuracy: {hyp.confidence:.2%})")


if __name__ == "__main__":
    # Run all demos
    demo_basic_learning()
    print("\n\n")
    demo_logical_operators()
    print("\n\n")
    demo_transform_library()
