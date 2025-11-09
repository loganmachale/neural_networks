"""
Demo script for the Logic Model (LM) Prototype

This demonstrates the reasoning capabilities of the model using
classic logical reasoning examples.
"""

from logic_model import (
    InternalState, Goal, LogicModel,
    AddFactDI, ApplyRuleDI
)


def demo_socrates_syllogism():
    """
    Classic syllogism:
    - All men are mortal
    - Socrates is a man
    - Therefore, Socrates is mortal

    This demonstrates how the model can use logical rules to derive conclusions.
    """
    print("=" * 70)
    print("DEMO 1: Socrates Syllogism")
    print("=" * 70)

    # Create initial state
    state = InternalState()

    # Add initial facts
    state.add_fact("socrates_is_man", True, 0.95)
    state.add_fact("all_men_mortal", True, 0.99)

    # Add logical rule: If X is a man, then X is mortal
    def mortality_rule(facts):
        socrates_is_man = facts.get("socrates_is_man")
        all_men_mortal = facts.get("all_men_mortal")

        if socrates_is_man and all_men_mortal:
            man_value, man_conf = socrates_is_man
            mortal_value, mortal_conf = all_men_mortal

            if man_value and mortal_value:
                # Derive conclusion with confidence as minimum of premises
                combined_conf = min(man_conf, mortal_conf)
                return ("socrates_is_mortal", True, combined_conf)
        return None

    state.add_rule(mortality_rule)

    # Define goal: Prove that Socrates is mortal
    goal = Goal({
        "socrates_is_mortal": (True, 0.9)
    })

    # Create and run the model
    lm = LogicModel(state, goal, search_depth=1)
    final_state = lm.think(max_iterations=5, verbose=True)

    print("\n" + "=" * 70)
    print("REASONING TRACE:")
    print("=" * 70)
    for step in lm.get_reasoning_trace():
        print(step)

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    mortal_fact = final_state.get_fact("socrates_is_mortal")
    if mortal_fact:
        value, confidence = mortal_fact
        print(f"Socrates is mortal: {value} (confidence: {confidence:.2f})")
    else:
        print("Could not derive conclusion")


def demo_weather_reasoning():
    """
    Weather reasoning scenario:
    - If the sky is cloudy and humidity is high, it will rain
    - The sky is cloudy
    - Goal: Determine if it will rain
    """
    print("\n\n" + "=" * 70)
    print("DEMO 2: Weather Reasoning")
    print("=" * 70)

    # Create initial state
    state = InternalState()

    # Add initial facts
    state.add_fact("sky_is_cloudy", True, 0.85)
    state.add_fact("humidity_high", True, 0.70)

    # Add weather rule
    def rain_rule(facts):
        cloudy = facts.get("sky_is_cloudy")
        humid = facts.get("humidity_high")

        if cloudy and humid:
            cloudy_val, cloudy_conf = cloudy
            humid_val, humid_conf = humid

            if cloudy_val and humid_val:
                # Derive rain prediction
                combined_conf = min(cloudy_conf, humid_conf) * 0.9
                return ("will_rain", True, combined_conf)
        return None

    state.add_rule(rain_rule)

    # Define goal: Predict rain
    goal = Goal({
        "will_rain": (True, 0.6)
    })

    # Create and run the model
    lm = LogicModel(state, goal, search_depth=1)
    final_state = lm.think(max_iterations=5, verbose=True)

    print("\n" + "=" * 70)
    print("REASONING TRACE:")
    print("=" * 70)
    for step in lm.get_reasoning_trace():
        print(step)

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    rain_fact = final_state.get_fact("will_rain")
    if rain_fact:
        value, confidence = rain_fact
        print(f"It will rain: {value} (confidence: {confidence:.2f})")
    else:
        print("Could not predict rain")


def demo_mathematical_reasoning():
    """
    Mathematical reasoning:
    - x = 5
    - y = 3
    - Goal: Determine if x > y
    """
    print("\n\n" + "=" * 70)
    print("DEMO 3: Mathematical Reasoning")
    print("=" * 70)

    # Create initial state
    state = InternalState()

    # Add initial facts
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
                return ("x_greater_than_y", is_greater, combined_conf)
        return None

    state.add_rule(comparison_rule)

    # Define goal: Prove x > y
    goal = Goal({
        "x_greater_than_y": (True, 0.95)
    })

    # Create and run the model
    lm = LogicModel(state, goal, search_depth=1)
    final_state = lm.think(max_iterations=5, verbose=True)

    print("\n" + "=" * 70)
    print("REASONING TRACE:")
    print("=" * 70)
    for step in lm.get_reasoning_trace():
        print(step)

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    comparison_fact = final_state.get_fact("x_greater_than_y")
    if comparison_fact:
        value, confidence = comparison_fact
        print(f"x > y: {value} (confidence: {confidence:.2f})")
    else:
        print("Could not determine comparison")


if __name__ == "__main__":
    # Run all demos
    demo_socrates_syllogism()
    demo_weather_reasoning()
    demo_mathematical_reasoning()

    print("\n\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nThe Logic Model successfully demonstrated:")
    print("1. Logical deduction (Socrates syllogism)")
    print("2. Causal reasoning (weather prediction)")
    print("3. Mathematical reasoning (numerical comparison)")
    print("\nKey features showcased:")
    print("- Tree search for optimal reasoning paths")
    print("- Confidence tracking throughout reasoning")
    print("- Logical rule application")
    print("- Goal-directed thinking")
