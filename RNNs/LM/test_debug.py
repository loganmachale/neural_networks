"""Quick debug test for demo 3"""

from logic_model import InternalState, Goal, ScoringModule, ApplyRuleDI

# Create initial state
state = InternalState()
state.add_fact("x_equals_5", 5, 1.0)
state.add_fact("y_equals_3", 3, 1.0)

# Add comparison rule
def comparison_rule(facts):
    x_fact = facts.get("x_equals_5")
    y_fact = facts.get("y_equals_3")

    if x_fact and y_fact:
        x_val, x_conf = x_fact
        y_val, y_conf = y_fact

        if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
            is_greater = x_val > y_val
            combined_conf = min(x_conf, y_conf)
            print(f"Rule evaluated: x={x_val}, y={y_val}, greater={is_greater}, conf={combined_conf}")
            return ("x_greater_than_y", is_greater, combined_conf)
    print("Rule returned None")
    return None

state.add_rule(comparison_rule)

# Define goal
goal = Goal({
    "x_greater_than_y": (True, 0.95)
})

# Test rule application
print("Initial state:", state)
print("Number of rules:", len(state.rules))

# Create scoring module
scorer = ScoringModule()
initial_score = scorer.score(state, goal)
print(f"Initial score: {initial_score:.2f}")

# Apply the rule
action = ApplyRuleDI(0)
new_state = action.apply(state)
print("\nAfter applying rule:", new_state)

new_score = scorer.score(new_state, goal)
print(f"New score: {new_score:.2f}")

print(f"\nScore improvement: {new_score - initial_score:.2f}")
