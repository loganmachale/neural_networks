"""
Demo script showing how to use the visualization tools with the Logic Model.

This script demonstrates tracking and visualizing metrics during reasoning.
"""

from logic_model import InternalState, Goal
from visualizations import (
    LogicModelWithTracking,
    plot_loss_vs_epochs,
    plot_score_vs_epochs,
    plot_combined_metrics,
    plot_confidence_vs_epochs,
    plot_action_distribution,
    plot_all_metrics,
    create_training_summary
)


def demo_complex_reasoning_with_visualization():
    """
    A more complex reasoning scenario that takes multiple steps,
    perfect for demonstrating the visualization capabilities.
    """
    print("=" * 70)
    print("VISUALIZATION DEMO: Multi-Step Reasoning Problem")
    print("=" * 70)
    print()
    print("Scenario: Medical Diagnosis")
    print("- Patient has symptoms: fever, cough")
    print("- Need to diagnose and recommend treatment")
    print("=" * 70)
    print()

    # Create initial state with symptoms
    state = InternalState()
    state.add_fact("has_fever", True, 0.85)
    state.add_fact("has_cough", True, 0.80)
    state.add_fact("has_headache", True, 0.60)

    # Add diagnostic rules
    def flu_diagnosis_rule(facts):
        """If fever AND cough, likely flu"""
        fever = facts.get("has_fever")
        cough = facts.get("has_cough")

        if fever and cough:
            fever_val, fever_conf = fever
            cough_val, cough_conf = cough

            if fever_val and cough_val:
                confidence = min(fever_conf, cough_conf) * 0.85
                return ("diagnosed_with_flu", True, confidence)
        return None

    def treatment_rule(facts):
        """If diagnosed with flu, recommend rest"""
        flu = facts.get("diagnosed_with_flu")

        if flu:
            flu_val, flu_conf = flu
            if flu_val:
                return ("needs_rest", True, flu_conf * 0.95)
        return None

    def medication_rule(facts):
        """If diagnosed with flu and fever, recommend medication"""
        flu = facts.get("diagnosed_with_flu")
        fever = facts.get("has_fever")

        if flu and fever:
            flu_val, flu_conf = flu
            fever_val, fever_conf = fever

            if flu_val and fever_val:
                confidence = min(flu_conf, fever_conf) * 0.90
                return ("needs_medication", True, confidence)
        return None

    state.add_rule(flu_diagnosis_rule)
    state.add_rule(treatment_rule)
    state.add_rule(medication_rule)

    # Define goal: reach diagnosis and treatment plan
    goal = Goal({
        "diagnosed_with_flu": (True, 0.6),
        "needs_rest": (True, 0.6),
        "needs_medication": (True, 0.6)
    })

    # Create model with tracking
    print("Creating Logic Model with metric tracking...")
    lm = LogicModelWithTracking(state, goal, search_depth=2)

    # Run the reasoning process
    print("Running reasoning process...\n")
    final_state = lm.think(max_iterations=15, verbose=True)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS:")
    print("=" * 70)

    diagnosis = final_state.get_fact("diagnosed_with_flu")
    rest = final_state.get_fact("needs_rest")
    medication = final_state.get_fact("needs_medication")

    if diagnosis:
        val, conf = diagnosis
        print(f"Flu diagnosis: {val} (confidence: {conf:.2%})")

    if rest:
        val, conf = rest
        print(f"Needs rest: {val} (confidence: {conf:.2%})")

    if medication:
        val, conf = medication
        print(f"Needs medication: {val} (confidence: {conf:.2%})")

    # Print training summary
    print("\n")
    print(create_training_summary(lm.metrics))

    # Generate all visualizations
    print("\nGenerating visualizations...")
    print("(Close each plot window to see the next one)\n")

    # Individual plots
    plot_score_vs_epochs(lm.metrics, title="Medical Diagnosis: Score vs Epochs")
    plot_loss_vs_epochs(lm.metrics, title="Medical Diagnosis: Loss vs Epochs")
    plot_combined_metrics(lm.metrics, title="Medical Diagnosis: Combined Metrics")
    plot_confidence_vs_epochs(lm.metrics, title="Medical Diagnosis: Confidence vs Epochs")
    plot_action_distribution(lm.metrics, title="Medical Diagnosis: Action Distribution")

    # Complete dashboard
    plot_all_metrics(lm.metrics, title_prefix="Medical Diagnosis - ")

    print("\nVisualization demo complete!")
    return lm


def demo_simple_with_tracking():
    """
    Simple Socrates example with tracking for quick visualization.
    """
    print("\n\n" + "=" * 70)
    print("SIMPLE VISUALIZATION DEMO: Socrates Syllogism")
    print("=" * 70)

    # Create initial state
    state = InternalState()
    state.add_fact("socrates_is_man", True, 0.95)
    state.add_fact("all_men_mortal", True, 0.99)

    # Add logical rule
    def mortality_rule(facts):
        socrates_is_man = facts.get("socrates_is_man")
        all_men_mortal = facts.get("all_men_mortal")

        if socrates_is_man and all_men_mortal:
            man_value, man_conf = socrates_is_man
            mortal_value, mortal_conf = all_men_mortal

            if man_value and mortal_value:
                combined_conf = min(man_conf, mortal_conf)
                return ("socrates_is_mortal", True, combined_conf)
        return None

    state.add_rule(mortality_rule)

    # Define goal
    goal = Goal({"socrates_is_mortal": (True, 0.9)})

    # Create and run model with tracking
    lm = LogicModelWithTracking(state, goal, search_depth=1)
    final_state = lm.think(max_iterations=5, verbose=True)

    # Show summary
    print("\n" + create_training_summary(lm.metrics))

    # Quick visualization
    if len(lm.metrics.epochs) > 0:
        plot_combined_metrics(lm.metrics, title="Socrates Syllogism: Metrics")

    return lm


if __name__ == "__main__":
    print("LOGIC MODEL VISUALIZATION DEMOS")
    print("=" * 70)
    print()
    print("This script demonstrates how to track and visualize metrics")
    print("during the reasoning process.")
    print()
    print("You will see:")
    print("  1. A complex multi-step medical diagnosis problem")
    print("  2. Score, loss, and confidence plots over time")
    print("  3. Action distribution analysis")
    print("  4. A comprehensive metrics dashboard")
    print()
    print("=" * 70)
    print()
    input("Press Enter to start the demo...")

    # Run the complex demo
    lm_complex = demo_complex_reasoning_with_visualization()

    # Optionally run simple demo
    print("\n\n")
    choice = input("Run simple Socrates demo? (y/n): ").strip().lower()
    if choice == 'y':
        lm_simple = demo_simple_with_tracking()

    print("\n" + "=" * 70)
    print("ALL VISUALIZATION DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - LogicModelWithTracking automatically records metrics")
    print("  - Multiple visualization functions available")
    print("  - plot_all_metrics() creates a comprehensive dashboard")
    print("  - Metrics include: score, loss, confidence, actions, state size")
