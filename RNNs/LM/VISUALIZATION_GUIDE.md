# Visualization Guide for Logic Model

This guide explains how to use the visualization tools to track and analyze your reasoning model's performance.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install matplotlib numpy
   ```
   Or use the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the visualization demo**:
   ```bash
   python demo_visualizations.py
   ```

## Available Plots

### 1. Loss vs Epochs
Shows how the loss (negative score) changes over time. Lower is better.

```python
from visualizations import plot_loss_vs_epochs

plot_loss_vs_epochs(lm.metrics, title="My Problem: Loss vs Epochs")
```

### 2. Score vs Epochs
Shows how the model's score improves over time. Higher is better.

```python
from visualizations import plot_score_vs_epochs

plot_score_vs_epochs(lm.metrics, title="My Problem: Score vs Epochs")
```

### 3. Combined Metrics (Score + Loss)
Shows both score and loss on the same plot with dual y-axes.

```python
from visualizations import plot_combined_metrics

plot_combined_metrics(lm.metrics, title="Training Progress")
```

### 4. Confidence vs Epochs
Tracks the average confidence level of all beliefs over time.

```python
from visualizations import plot_confidence_vs_epochs

plot_confidence_vs_epochs(lm.metrics, title="Confidence Tracking")
```

### 5. Action Distribution
Bar chart showing which types of actions the model used most frequently.

```python
from visualizations import plot_action_distribution

plot_action_distribution(lm.metrics, title="Action Usage")
```

### 6. Complete Dashboard
Generates all visualizations in a single comprehensive dashboard.

```python
from visualizations import plot_all_metrics

plot_all_metrics(lm.metrics, title_prefix="My Problem - ")
```

## Using Metric Tracking

### Step 1: Import the tracking model

```python
from visualizations import LogicModelWithTracking
from logic_model import InternalState, Goal
```

### Step 2: Create your problem

```python
# Create initial state
state = InternalState()
state.add_fact("premise_1", True, 0.9)
state.add_fact("premise_2", True, 0.85)

# Add rules
def my_rule(facts):
    # Your rule logic here
    return ("conclusion", True, 0.8)

state.add_rule(my_rule)

# Define goal
goal = Goal({"conclusion": (True, 0.7)})
```

### Step 3: Create model with tracking

```python
# Use LogicModelWithTracking instead of LogicModel
lm = LogicModelWithTracking(state, goal, search_depth=2)
```

### Step 4: Run reasoning

```python
final_state = lm.think(max_iterations=10, verbose=True)
```

### Step 5: Visualize results

```python
from visualizations import plot_all_metrics, create_training_summary

# Print summary
print(create_training_summary(lm.metrics))

# Generate visualizations
plot_all_metrics(lm.metrics, title_prefix="My Problem - ")
```

## Saving Plots

All plot functions accept a `save_path` parameter:

```python
plot_score_vs_epochs(lm.metrics,
                    title="Training Progress",
                    save_path="results/score_plot.png")
```

## Metrics Tracked Automatically

When using `LogicModelWithTracking`, the following metrics are recorded at each step:

- **Epoch number**: The current iteration
- **Score**: The evaluation score of the current state
- **Loss**: Negative of the score (for minimization perspective)
- **Average confidence**: Mean confidence across all facts
- **Action type**: Which type of action was taken
- **State size**: Number of facts in the belief state

## Example: Complete Workflow

```python
from logic_model import InternalState, Goal
from visualizations import (
    LogicModelWithTracking,
    plot_all_metrics,
    create_training_summary
)

# 1. Set up problem
state = InternalState()
state.add_fact("x", 5, 1.0)
state.add_fact("y", 3, 1.0)

def comparison_rule(facts):
    x = facts.get("x")
    y = facts.get("y")
    if x and y:
        x_val, x_conf = x
        y_val, y_conf = y
        return ("x_greater_than_y", x_val > y_val, min(x_conf, y_conf))
    return None

state.add_rule(comparison_rule)
goal = Goal({"x_greater_than_y": (True, 0.9)})

# 2. Create and run model
lm = LogicModelWithTracking(state, goal, search_depth=1)
final_state = lm.think(max_iterations=5, verbose=True)

# 3. Analyze results
print(create_training_summary(lm.metrics))
plot_all_metrics(lm.metrics, title_prefix="Comparison Problem - ")
```

## Tips

1. **For quick problems** (1-2 steps): Individual plots are sufficient
2. **For complex problems** (5+ steps): Use `plot_all_metrics()` for comprehensive view
3. **Set search_depth** appropriately: Higher depth = better reasoning but slower
4. **Monitor confidence**: Declining confidence may indicate contradictions
5. **Action distribution**: Shows if the model is stuck using only certain actions

## Interpreting Results

### Good Training Progress
- Score increases over epochs
- Loss decreases over epochs
- Confidence remains stable or increases
- Diverse action distribution

### Potential Issues
- Score plateaus early → May need more iterations or better rules
- Confidence drops → Check for contradictions in rules
- Only one action type used → Action space may be too limited
- Score decreases → Adversarial actions dominating (increase search depth)

## Advanced: Custom Metrics

You can extend `MetricsTracker` to track custom metrics:

```python
from visualizations import MetricsTracker

class CustomTracker(MetricsTracker):
    def __init__(self):
        super().__init__()
        self.custom_metric = []

    def record_custom(self, value):
        self.custom_metric.append(value)
```

---

For more examples, see `demo_visualizations.py`.
