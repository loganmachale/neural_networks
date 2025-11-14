# ARC AGI Challenge - Logic Model Approach

A reasoning system for solving the Abstraction and Reasoning Corpus (ARC) challenge using tree search and logical inference.

## Overview

This project applies a logic model approach to the ARC AGI challenge, using:
- **Tree Search**: Minimax with Alpha-Beta pruning + MCTS
- **Internal State Representation**: Beliefs about grid patterns and transformations
- **Goal-Directed Reasoning**: Search for transformation rules that match examples
- **Confidence Tracking**: Probabilistic reasoning about pattern hypotheses

## ARC Challenge Data

The repository includes:
- **Training**: 1000 ARC problems in `data/training/`
- **Evaluation**: 120 ARC problems in `data/evaluation/`

Each problem consists of:
- Input/output grid examples (training pairs)
- Test input(s) requiring prediction

### Data Format

```json
{
  "train": [
    {
      "input": [[7, 9], [4, 3]],
      "output": [[7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3], ...]
    }
  ],
  "test": [
    {
      "input": [[3, 2], [7, 8]],
      "output": [[3, 2, 3, 2, 3, 2], ...]
    }
  ]
}
```

## Core Architecture

### 1. Logic Model Engine (`logic_model.py`)

**Core Components:**
- `InternalState`: Represents beliefs about grid patterns, transformations, and rules
- `ActionDI`: Internal actions for hypothesis generation and refinement
- `Goal`: Target state representing solved problem
- `ScoringModule`: Evaluates quality of pattern hypotheses
- `SearchModule`: Tree search (Alpha-Beta + MCTS) for optimal reasoning paths

**Search Algorithms:**
- **Alpha-Beta Pruning**: Fast, deterministic search for clear patterns
- **MCTS**: Probabilistic search for complex, ambiguous patterns

### 2. Visualization Tools (`visualizations.py`)

Tracks and visualizes the reasoning process:
- Score progression over iterations
- Confidence in pattern hypotheses
- Action distribution
- Search tree statistics

### 3. Advanced Features (`advanced_features.py`)

Experimental learning components:
- Neural network scorer stubs (for future learning)
- Replay buffer for experience storage
- Self-play training framework

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `matplotlib` - Visualization
- `numpy` - Numerical operations

## Quick Start

### Load ARC Data

```python
import json

# Load a training problem
with open('data/training/00576224.json', 'r') as f:
    problem = json.load(f)

train_examples = problem['train']
test_cases = problem['test']

print(f"Training examples: {len(train_examples)}")
print(f"Test cases: {len(test_cases)}")
```

### Basic Reasoning Example

```python
from logic_model import InternalState, Goal, LogicModel

# Create initial state (empty hypothesis space)
state = InternalState()

# Add observations from first training example
state.add_fact("input_shape", (2, 2), 1.0)
state.add_fact("output_shape", (6, 6), 1.0)
state.add_fact("pattern_type", "repetition", 0.7)

# Define pattern recognition rules
def repetition_rule(facts):
    """Hypothesize grid repetition pattern."""
    if facts.get("pattern_type"):
        pattern_val, pattern_conf = facts.get("pattern_type")
        if pattern_val == "repetition":
            return ("transformation", "repeat_3x3", pattern_conf * 0.9)
    return None

state.add_rule(repetition_rule)

# Define goal: identify transformation
goal = Goal({
    "transformation": ("repeat_3x3", 0.8)
})

# Run reasoning
lm = LogicModel(state, goal, search_type='alphabeta', search_depth=3)
final_state = lm.think(max_iterations=10, verbose=True)

# Check result
transformation = final_state.get_fact("transformation")
print(f"Discovered transformation: {transformation}")
```

## Implementation Status

### ✅ Phase 1: Core Infrastructure (COMPLETE)

**Implemented Components:**
1. ✅ **ARC Data Loader** (`arc_loader.py`)
   - Loads training and evaluation problems
   - Parses JSON format
   - Provides visualization utilities

2. ✅ **Grid Primitives** (`grid_primitives.py`)
   - Grid representation with numpy backend
   - Cell queries and spatial operations
   - Pattern detection helpers
   - **NO hard-coded transformations** - only data structures

3. ✅ **Logical Operators** (`logical_operators.py`)
   - **Hard-coded primitives**: AND, OR, NOT, IF, ==
   - Cell predicates (cell_eq, count_gt, etc.)
   - Conditional transforms (IF-THEN-ELSE)
   - Composite transforms (sequential composition)
   - **These are the language primitives, not learned**

4. ✅ **Transform Learning** (`transform_learning.py`)
   - Learns transformations from examples
   - Generates hypotheses from input/output pairs
   - Tests hypotheses for consistency
   - Builds reusable transform library
   - **Current patterns detected:**
     - Color mapping
     - Tiling with checkerboard patterns
     - Scaling
     - Conditional rules

**Demonstration:**
```bash
python demo_arc.py
```

Results on problem `00576224`:
- **Pattern detected**: Checkerboard tile 3x3 (horizontal flip alternating)
- **Training accuracy**: 100% (2/2 examples)
- **Test accuracy**: 100% (predicted output matches expected)

### Phase 2: Advanced Pattern Learning (Next)

Expand pattern detection capabilities:
1. Rotation and reflection detection
2. Object detection and tracking
3. Spatial relationship rules
4. Multi-step transformations
5. Transfer learning across similar problems

### Phase 3: Search Integration

Integrate logic model for complex reasoning:
1. Use tree search to explore transformation compositions
2. MCTS for hypothesis space exploration
3. Goal-directed reasoning for multi-step solutions
4. Confidence tracking for uncertain patterns

### Phase 4: Neural Integration (Future)

Replace heuristics with learned models:
1. Neural network for pattern recognition
2. Learned value function for hypothesis scoring
3. Policy network for action selection
4. Self-supervised learning on ARC corpus

## Key Features

### Core Capabilities
1. **Tree Search**: Alpha-Beta + MCTS for hypothesis exploration
2. **Confidence Tracking**: Probabilistic reasoning
3. **Rule-Based Inference**: Logical pattern composition
4. **Goal-Directed**: Optimizes toward transformation discovery

### Performance Optimizations
5. **State Hashing**: Efficient hypothesis deduplication
6. **Memoization**: Caches evaluated states
7. **Early Termination**: Stops when pattern found
8. **Cache Statistics**: Performance monitoring

### Advanced Features
9. **MCTS Search**: Alternative to Alpha-Beta for complex patterns
10. **Neural Stubs**: Framework for future learning
11. **Replay Buffer**: Experience storage
12. **Visualization**: Comprehensive metrics tracking

## Architecture

The reasoning loop:
1. **Project**: Generate pattern hypotheses using tree search
2. **Assess**: Evaluate hypotheses against training examples
3. **Iterate**: Refine hypotheses and repeat

Scoring function evaluates hypotheses based on:
- **Consistency**: Matches all training examples
- **Simplicity**: Occam's razor (simpler patterns preferred)
- **Confidence**: Statistical strength of pattern
- **Generality**: Likely to work on test cases

## Testing

```bash
# Run core tests
python test_logic_model.py
```

Tests verify:
- State hashing and equality
- Goal achievement detection
- Search algorithms (Alpha-Beta + MCTS)
- Rule application
- Confidence propagation

## Project Structure

```
.
├── arc_loader.py              # ARC data loader and visualization
├── grid_primitives.py         # Grid data structures (no hard-coded transforms)
├── logical_operators.py       # Hard-coded logical primitives (AND, OR, IF, ==)
├── transform_learning.py      # Transform learning system (learns from examples)
├── demo_arc.py               # Demonstration script
│
├── logic_model.py             # Core reasoning engine (tree search)
├── visualizations.py          # Metrics and plotting
├── advanced_features.py       # Neural network stubs
├── test_logic_model.py        # Test suite
│
├── requirements.txt           # Dependencies
├── data/
│   ├── training/             # 1000 ARC training problems
│   └── evaluation/           # 120 ARC evaluation problems
└── README.md                  # This file
```

**Core Philosophy:**
- **Hard-coded**: Logical operators (AND, OR, IF, ==) - the language primitives
- **Learned**: Grid transformations - discovered from examples
- **Composable**: Learned transforms can be combined with logical operators

## ARC Challenge Strategy

### Pattern Categories to Detect

1. **Geometric Transformations**
   - Rotation, reflection, translation
   - Scaling, tiling, repetition

2. **Color Operations**
   - Palette mapping
   - Conditional recoloring
   - Color arithmetic

3. **Object Operations**
   - Object detection and tracking
   - Object completion
   - Object composition

4. **Logical Rules**
   - If-then patterns
   - Counting-based rules
   - Spatial relationships

### Evaluation Metrics

Success measured by:
- **Exact Match**: Predicted grid = expected output
- **Pattern Coverage**: % of problems where pattern detected
- **Generalization**: Performance on evaluation set
- **Efficiency**: Iterations/time to solution

## Next Steps

### Completed ✅
1. ✅ **ARC data loader** - Loads and parses JSON problems
2. ✅ **Grid representation primitives** - Data structures without hard-coded transforms
3. ✅ **Hard-coded logical operators** - AND, OR, IF, ==, etc.
4. ✅ **Transform learning system** - Learns from examples
5. ✅ **Test on simple problems** - Successfully solved checkerboard tiling pattern

### In Progress 🔄
6. **Expand pattern library**:
   - Add rotation/reflection detection
   - Add object detection
   - Add symmetry detection
   - Add fill/flood patterns

### Upcoming 📋
7. **Integrate tree search** - Use logic model for multi-step transformations
8. **Composition system** - Combine learned transforms with logical operators
9. **Evaluation suite** - Systematic testing on ARC benchmark
10. **Advanced patterns** - Handle more complex transformation types

## Resources

- [ARC Challenge](https://github.com/fchollet/ARC-AGI) - Official repository
- [ARC Paper](https://arxiv.org/abs/1911.01547) - "On the Measure of Intelligence"
- Training data: `data/training/` (1000 problems)
- Evaluation data: `data/evaluation/` (120 problems)

## License

Research and educational use.
