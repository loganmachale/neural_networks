"""
Logic Model Examples and Demonstrations

This module contains all example demonstrations for the Logic Model:
1. Basic reasoning demos (Socrates, weather, math)
2. Visualization demos (medical diagnosis, tracked reasoning)
3. Addition learning demo (trial-and-error learning with ML evaluation)

Usage:
    python examples.py                    # Run all demos
    python examples.py --basic            # Run only basic demos
    python examples.py --viz              # Run only visualization demos
    python examples.py --addition         # Run only addition learning demo
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from logic_model import (
    InternalState, Goal, LogicModel,
    AddFactDI, ApplyRuleDI, ActionDI, ScoringModule
)
from visualizations import (
    LogicModelWithTracking, plot_loss_vs_epochs, plot_score_vs_epochs,
    plot_combined_metrics, plot_confidence_vs_epochs, plot_action_distribution,
    plot_all_metrics, create_training_summary
)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# Addition learning configuration
MAX_NUMBER = 20
MIN_NUMBER = 0
MAX_ATTEMPTS_PER_PROBLEM = 10
NUM_TRAINING_EPISODES = 50
VALIDATION_SET_SIZE = 20
TEST_SET_SIZE = 10
EVAL_FREQUENCY = 5


# =============================================================================
# BASIC REASONING DEMOS
# =============================================================================

def demo_socrates_syllogism():
    """
    Classic syllogism demonstrating logical deduction.

    Premises:
        - All men are mortal
        - Socrates is a man
    Conclusion:
        - Therefore, Socrates is mortal
    """
    print("=" * 70)
    print("DEMO 1: Socrates Syllogism")
    print("=" * 70)

    # Create initial state
    state = InternalState()
    state.add_fact("socrates_is_man", True, 0.95)
    state.add_fact("all_men_mortal", True, 0.99)

    # Add logical rule
    def mortality_rule(facts):
        """Apply modus ponens: if X is a man and all men are mortal, then X is mortal."""
        socrates_is_man = facts.get("socrates_is_man")
        all_men_mortal = facts.get("all_men_mortal")

        if socrates_is_man and all_men_mortal:
            man_value, man_conf = socrates_is_man
            mortal_value, mortal_conf = all_men_mortal

            if man_value and mortal_value:
                # Derive conclusion with minimum confidence of premises
                combined_conf = min(man_conf, mortal_conf)
                return ("socrates_is_mortal", True, combined_conf)
        return None

    state.add_rule(mortality_rule)

    # Define goal
    goal = Goal({
        "socrates_is_mortal": (True, 0.9)
    })

    # Run model
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
    Weather prediction demonstrating causal reasoning.

    Rules:
        - If sky is cloudy AND humidity is high, then it will rain
    Given:
        - Sky is cloudy
        - Humidity is high
    Predict:
        - Will it rain?
    """
    print("\n\n" + "=" * 70)
    print("DEMO 2: Weather Reasoning")
    print("=" * 70)

    # Create initial state
    state = InternalState()
    state.add_fact("sky_is_cloudy", True, 0.85)
    state.add_fact("humidity_high", True, 0.70)

    # Add weather rule
    def rain_rule(facts):
        """Predict rain based on cloud cover and humidity."""
        cloudy = facts.get("sky_is_cloudy")
        humid = facts.get("humidity_high")

        if cloudy and humid:
            cloudy_val, cloudy_conf = cloudy
            humid_val, humid_conf = humid

            if cloudy_val and humid_val:
                # Rain prediction with reduced confidence
                combined_conf = min(cloudy_conf, humid_conf) * 0.9
                return ("will_rain", True, combined_conf)
        return None

    state.add_rule(rain_rule)

    # Define goal
    goal = Goal({
        "will_rain": (True, 0.6)
    })

    # Run model
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
    Numerical comparison demonstrating mathematical reasoning.

    Given:
        - x = 5
        - y = 3
    Determine:
        - Is x > y?
    """
    print("\n\n" + "=" * 70)
    print("DEMO 3: Mathematical Reasoning")
    print("=" * 70)

    # Create initial state
    state = InternalState()
    state.add_fact("x_equals_5", 5, 1.0)
    state.add_fact("y_equals_3", 3, 1.0)

    # Add comparison rule
    def comparison_rule(facts):
        """Compare two numerical values."""
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

    # Define goal
    goal = Goal({
        "x_greater_than_y": (True, 0.95)
    })

    # Run model
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


def run_basic_demos():
    """Run all basic reasoning demonstrations."""
    demo_socrates_syllogism()
    demo_weather_reasoning()
    demo_mathematical_reasoning()

    print("\n\n" + "=" * 70)
    print("ALL BASIC DEMOS COMPLETE")
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


# =============================================================================
# VISUALIZATION DEMOS
# =============================================================================

def demo_medical_diagnosis_visualization():
    """
    Multi-step medical diagnosis demonstrating visualization capabilities.

    Scenario:
        - Patient symptoms: fever, cough, headache
        - Diagnose condition
        - Recommend treatment
    """
    print("=" * 70)
    print("VISUALIZATION DEMO: Multi-Step Medical Diagnosis")
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
        """Diagnose flu based on fever and cough."""
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
        """Recommend rest if diagnosed with flu."""
        flu = facts.get("diagnosed_with_flu")

        if flu:
            flu_val, flu_conf = flu
            if flu_val:
                return ("needs_rest", True, flu_conf * 0.95)
        return None

    def medication_rule(facts):
        """Recommend medication for flu with fever."""
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

    # Define goal
    goal = Goal({
        "diagnosed_with_flu": (True, 0.6),
        "needs_rest": (True, 0.6),
        "needs_medication": (True, 0.6)
    })

    # Create model with tracking
    print("Creating Logic Model with metric tracking...")
    lm = LogicModelWithTracking(state, goal, search_depth=2)

    # Run reasoning
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

    # Generate visualizations
    print("\nGenerating visualizations...")
    print("(Close each plot window to see the next one)\n")

    plot_score_vs_epochs(lm.metrics, title="Medical Diagnosis: Score vs Epochs")
    plot_loss_vs_epochs(lm.metrics, title="Medical Diagnosis: Loss vs Epochs")
    plot_combined_metrics(lm.metrics, title="Medical Diagnosis: Combined Metrics")
    plot_confidence_vs_epochs(lm.metrics, title="Medical Diagnosis: Confidence vs Epochs")
    plot_action_distribution(lm.metrics, title="Medical Diagnosis: Action Distribution")
    plot_all_metrics(lm.metrics, title_prefix="Medical Diagnosis - ")

    print("\nVisualization demo complete!")
    return lm


def demo_socrates_with_tracking():
    """
    Simple Socrates example with metric tracking for visualization.
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
        """Apply modus ponens for mortality inference."""
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

    # Run model with tracking
    lm = LogicModelWithTracking(state, goal, search_depth=1)
    final_state = lm.think(max_iterations=5, verbose=True)

    # Show summary
    print("\n" + create_training_summary(lm.metrics))

    # Visualization
    if len(lm.metrics.epochs) > 0:
        plot_combined_metrics(lm.metrics, title="Socrates Syllogism: Metrics")

    return lm


def run_visualization_demos():
    """Run all visualization demonstrations."""
    print("LOGIC MODEL VISUALIZATION DEMOS")
    print("=" * 70)
    print()
    print("Demonstrating metric tracking and visualization capabilities")
    print()
    print("You will see:")
    print("  1. A complex multi-step medical diagnosis problem")
    print("  2. Score, loss, and confidence plots over time")
    print("  3. Action distribution analysis")
    print("  4. A comprehensive metrics dashboard")
    print()
    print("=" * 70)
    print()

    # Run complex demo
    lm_complex = demo_medical_diagnosis_visualization()

    # Run simple demo
    print("\n\n")
    lm_simple = demo_socrates_with_tracking()

    print("\n" + "=" * 70)
    print("ALL VISUALIZATION DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - LogicModelWithTracking automatically records metrics")
    print("  - Multiple visualization functions available")
    print("  - plot_all_metrics() creates a comprehensive dashboard")
    print("  - Metrics include: score, loss, confidence, actions, state size")


# =============================================================================
# ADDITION LEARNING DEMO - ENVIRONMENT
# =============================================================================

class AdditionEnvironment:
    """
    Environment for testing addition learning.

    Generates addition problems and provides feedback on answers.
    """

    def __init__(self, max_number: int = MAX_NUMBER, min_number: int = MIN_NUMBER):
        """
        Initialize addition environment.

        Args:
            max_number: Maximum value for operands
            min_number: Minimum value for operands
        """
        self.max_number = max_number
        self.min_number = min_number

    def generate_problem(self) -> Tuple[int, int]:
        """Generate random addition problem."""
        a = random.randint(self.min_number, self.max_number)
        b = random.randint(self.min_number, self.max_number)
        return (a, b)

    def check_answer(self, a: int, b: int, guess: int) -> Dict:
        """
        Check if guess is correct and provide feedback.

        Args:
            a: First operand
            b: Second operand
            guess: Proposed sum

        Returns:
            Dictionary with status, correct_answer, and error
        """
        correct_answer = a + b
        error = abs(guess - correct_answer)

        if guess == correct_answer:
            status = 'correct'
        elif guess > correct_answer:
            status = 'too_high'
        else:
            status = 'too_low'

        return {
            'status': status,
            'correct_answer': correct_answer,
            'error': error
        }


# =============================================================================
# ADDITION LEARNING DEMO - SCORING MODULE
# =============================================================================

class LearningScoringModule:
    """
    Adaptive scoring module that learns from feedback.

    Tracks known addition facts and performance statistics.
    """

    def __init__(self):
        """Initialize empty knowledge base."""
        self.known_sums: Dict[Tuple[int, int], int] = {}
        self.feedback_history: List[Dict] = []

    def process_attempt(self, guess: int, actual: int, problem: Tuple[int, int]) -> Dict:
        """
        Record attempt and update statistics.

        Args:
            guess: Proposed answer
            actual: Correct answer
            problem: The (a, b) tuple

        Returns:
            Attempt statistics
        """
        a, b = problem
        is_correct = (guess == actual)
        error = abs(guess - actual)

        self.feedback_history.append({
            'guess': guess,
            'actual': actual,
            'error': error,
            'correct': is_correct
        })

        return {
            'correct': is_correct,
            'error': error
        }

    def learn_from_success(self, problem: Tuple[int, int], answer: int):
        """
        Store successful result in knowledge base.

        Args:
            problem: The (a, b) tuple
            answer: Correct sum
        """
        a, b = problem
        # Store both orderings (commutativity)
        self.known_sums[(a, b)] = answer
        self.known_sums[(b, a)] = answer

    def get_statistics(self) -> Dict:
        """Calculate performance statistics."""
        if not self.feedback_history:
            return {
                'total_attempts': 0,
                'correct_attempts': 0,
                'accuracy': 0.0,
                'average_error': 0.0,
                'knowledge_base_size': 0
            }

        total = len(self.feedback_history)
        correct = len([f for f in self.feedback_history if f['correct']])
        avg_error = np.mean([f['error'] for f in self.feedback_history])

        return {
            'total_attempts': total,
            'correct_attempts': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'average_error': float(avg_error),
            'knowledge_base_size': len(self.known_sums) // 2
        }


# =============================================================================
# ADDITION LEARNING DEMO - LEARNER
# =============================================================================

class AdditionLearner:
    """
    Agent that learns addition through trial and error.

    Uses pattern matching and binary search refinement to improve guesses.
    """

    def __init__(self, environment: AdditionEnvironment):
        """
        Initialize learner.

        Args:
            environment: Addition environment for generating problems
        """
        self.env = environment
        self.scorer = LearningScoringModule()
        self.training_history: List[Dict] = []

    def make_initial_guess(self, a: int, b: int) -> int:
        """
        Make initial guess using learned patterns.

        Strategy:
            1. Check for exact match in knowledge base
            2. Check commutative match
            3. Use pattern matching from similar problems
            4. Random guess if no knowledge

        Args:
            a: First operand
            b: Second operand

        Returns:
            Initial guess
        """
        # Check exact match
        if (a, b) in self.scorer.known_sums:
            return self.scorer.known_sums[(a, b)]

        # Check commutative match
        if (b, a) in self.scorer.known_sums:
            return self.scorer.known_sums[(b, a)]

        # Pattern matching from similar problems
        if len(self.scorer.known_sums) > 5:
            similar_sums = [v for (ka, kb), v in self.scorer.known_sums.items()
                          if abs(ka - a) <= 2 and abs(kb - b) <= 2]
            if similar_sums:
                avg = sum(similar_sums) // len(similar_sums)
                return max(a, b, avg + random.randint(-3, 3))

        # Random guess
        return random.randint(max(0, a + b - 5), a + b + 5)

    def refine_guess(self, a: int, b: int, current_guess: int, feedback: Dict) -> int:
        """
        Refine guess using binary search based on feedback.

        Args:
            a: First operand
            b: Second operand
            current_guess: Previous guess
            feedback: Feedback dict with status and correct_answer

        Returns:
            Refined guess
        """
        correct_answer = feedback['correct_answer']

        if feedback['status'] == 'too_high':
            # Guess was too high, reduce by half the error
            adjustment = -(abs(current_guess - correct_answer) // 2 + 1)
            new_guess = max(0, current_guess + adjustment)
        else:  # too_low
            # Guess was too low, increase by half the error
            adjustment = abs(correct_answer - current_guess) // 2 + 1
            new_guess = current_guess + adjustment

        return new_guess

    def train(self, num_episodes: int = NUM_TRAINING_EPISODES,
              validation_size: int = VALIDATION_SET_SIZE,
              eval_frequency: int = EVAL_FREQUENCY,
              verbose: bool = True) -> Dict:
        """
        Train learner on addition problems with train/val split.

        Args:
            num_episodes: Number of training problems
            validation_size: Size of validation set
            eval_frequency: How often to evaluate on validation set
            verbose: Print progress

        Returns:
            Training statistics
        """
        # Generate fixed validation set
        random.seed(42)
        validation_set = [self.env.generate_problem() for _ in range(validation_size)]
        random.seed()  # Reset seed

        if verbose:
            print(f"\nCreated validation set: {validation_size} problems")
            print(f"Validation problems: {validation_set[:5]}... (showing first 5)\n")

        training_accuracy_history = []
        validation_accuracy_history = []
        training_error_history = []
        validation_error_history = []
        success_rate_history = []

        # Track rolling success rate
        recent_successes = []

        for episode in range(1, num_episodes + 1):
            # Generate training problem
            problem = self.env.generate_problem()
            a, b = problem

            # Make initial guess
            guess = self.make_initial_guess(a, b)

            # Get feedback
            feedback = self.env.check_answer(a, b, guess)
            self.scorer.process_attempt(guess, feedback['correct_answer'], problem)

            # Refine until correct or max attempts
            attempts = 1
            while feedback['status'] != 'correct' and attempts < MAX_ATTEMPTS_PER_PROBLEM:
                guess = self.refine_guess(a, b, guess, feedback)
                feedback = self.env.check_answer(a, b, guess)
                self.scorer.process_attempt(guess, feedback['correct_answer'], problem)
                attempts += 1

            # Learn if successful
            if feedback['status'] == 'correct':
                self.scorer.learn_from_success(problem, feedback['correct_answer'])
                recent_successes.append(1)
            else:
                recent_successes.append(0)

            # Keep only last 10 episodes for rolling average
            if len(recent_successes) > 10:
                recent_successes.pop(0)

            # Calculate training statistics
            train_stats = self.scorer.get_statistics()
            training_accuracy_history.append(train_stats['accuracy'])
            training_error_history.append(train_stats['average_error'])
            success_rate_history.append(np.mean(recent_successes))

            # Periodic validation evaluation
            if episode % eval_frequency == 0:
                val_acc, val_err = self.evaluate_on_set(validation_set)
                validation_accuracy_history.append(val_acc)
                validation_error_history.append(val_err)

                if verbose:
                    print(f"\nEpisode {episode}/{num_episodes}")
                    print(f"  Recent problem: {a} + {b} = {feedback['correct_answer']}")
                    print(f"  Success: {feedback['status'] == 'correct'} (in {attempts} attempts)")
                    print(f"  Training accuracy: {train_stats['accuracy']*100:.2f}%")
                    print(f"  Training avg error: {train_stats['average_error']:.2f}")
                    print(f"  Validation accuracy: {val_acc*100:.2f}%")
                    print(f"  Validation avg error: {val_err:.2f}")

        # Final evaluation
        final_train_stats = self.scorer.get_statistics()
        final_val_acc, final_val_err = self.evaluate_on_set(validation_set)

        # Check for overfitting
        train_val_gap = final_train_stats['accuracy'] - final_val_acc
        if train_val_gap > 0.15:
            print("\n[WARNING] Possible overfitting detected (train-val gap > 0.15)")
        elif train_val_gap < 0:
            print("\n[EXCELLENT] Good generalization: Validation >= Training")
        else:
            print("\n[OK] Healthy train-val gap")

        # Store results
        results = {
            'training_accuracy': training_accuracy_history,
            'validation_accuracy': validation_accuracy_history,
            'training_error': training_error_history,
            'validation_error': validation_error_history,
            'success_rate': success_rate_history,
            'final_train_accuracy': final_train_stats['accuracy'],
            'final_val_accuracy': final_val_acc,
            'final_train_error': final_train_stats['average_error'],
            'final_val_error': final_val_err,
            'knowledge_base_size': final_train_stats['knowledge_base_size']
        }

        return results

    def evaluate_on_set(self, problem_set: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        Evaluate learner on a set of problems without learning.

        Args:
            problem_set: List of (a, b) problems

        Returns:
            Tuple of (accuracy, average_error)
        """
        correct = 0
        errors = []

        for a, b in problem_set:
            guess = self.make_initial_guess(a, b)
            feedback = self.env.check_answer(a, b, guess)
            if feedback['status'] == 'correct':
                correct += 1
            errors.append(feedback['error'])

        accuracy = correct / len(problem_set) if problem_set else 0.0
        avg_error = np.mean(errors) if errors else 0.0

        return accuracy, float(avg_error)

    def test(self, num_problems: int = TEST_SET_SIZE) -> Dict:
        """
        Test learner on new problems without learning.

        Args:
            num_problems: Number of test problems

        Returns:
            Test statistics
        """
        test_results = []

        for i in range(1, num_problems + 1):
            problem = self.env.generate_problem()
            a, b = problem

            # Make guess (no refinement, just first guess)
            guess = self.make_initial_guess(a, b)
            feedback = self.env.check_answer(a, b, guess)

            test_results.append({
                'problem': problem,
                'guess': guess,
                'correct_answer': feedback['correct_answer'],
                'correct': feedback['status'] == 'correct',
                'error': feedback['error']
            })

            print(f"\nTest {i}: {a} + {b} = ?")
            print(f"  Model's answer: {guess}")
            print(f"  Correct answer: {feedback['correct_answer']}")
            print(f"  Result: [{'CORRECT' if feedback['status'] == 'correct' else 'INCORRECT'}]")

        # Calculate statistics
        num_correct = sum(1 for r in test_results if r['correct'])
        accuracy = num_correct / len(test_results) if test_results else 0.0
        avg_error = np.mean([r['error'] for r in test_results])

        print(f"\nTest Accuracy: {num_correct}/{len(test_results)} = {accuracy*100:.1f}%")

        return {
            'accuracy': accuracy,
            'average_error': float(avg_error),
            'results': test_results
        }


# =============================================================================
# ADDITION LEARNING DEMO - VISUALIZATION
# =============================================================================

def visualize_addition_learning(results: Dict, filename: str = "addition_learning_progress.png"):
    """
    Create 4-panel visualization of addition learning progress.

    Args:
        results: Training results dictionary
        filename: Output filename for plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: Training vs Validation Accuracy
    ax1 = axes[0, 0]
    train_episodes = list(range(1, len(results['training_accuracy']) + 1))
    val_episodes = list(range(EVAL_FREQUENCY, len(results['training_accuracy']) + 1, EVAL_FREQUENCY))

    ax1.plot(train_episodes, results['training_accuracy'], 'g-', label='Training Accuracy', alpha=0.7)
    ax1.plot(val_episodes, results['validation_accuracy'], 'b-o', label='Validation Accuracy', markersize=6)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (100%)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Training vs Validation Error
    ax2 = axes[0, 1]
    ax2.plot(train_episodes, results['training_error'], 'r-', label='Training Error', alpha=0.7)
    ax2.plot(val_episodes, results['validation_error'], 'orange', marker='s',
             label='Validation Error', markersize=6)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Error')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Error')
    ax2.set_title('Training vs Validation Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Success Rate (Rolling Average)
    ax3 = axes[1, 0]
    ax3.plot(train_episodes, results['success_rate'], 'b-', label='Success Rate (window=10)')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Success Rate (Rolling Average, Window=10)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Overfitting Detection (Train - Val Gap)
    ax4 = axes[1, 1]
    gaps = []
    for i in range(len(results['validation_accuracy'])):
        train_idx = (i + 1) * EVAL_FREQUENCY - 1
        if train_idx < len(results['training_accuracy']):
            gap = results['training_accuracy'][train_idx] - results['validation_accuracy'][i]
            gaps.append(gap)

    gap_episodes = list(range(EVAL_FREQUENCY, EVAL_FREQUENCY * len(gaps) + 1, EVAL_FREQUENCY))
    ax4.plot(gap_episodes, gaps, 'purple', marker='D', label='Train - Val Gap')
    ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No Gap (Perfect Generalization)')
    ax4.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Accuracy Gap')
    ax4.set_title('Overfitting Detection (Train - Validation Gap)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {filename}")
    plt.close()


def run_addition_learning_demo():
    """Run complete addition learning demonstration."""
    print("=" * 70)
    print("ADDITION LEARNING DEMO")
    print("=" * 70)
    print()
    print("Demonstrating trial-and-error learning with proper ML evaluation")
    print()
    print("The model will:")
    print("  1. Start with no knowledge of arithmetic")
    print("  2. Make guesses and receive feedback")
    print("  3. Learn from mistakes and improve")
    print("  4. Be evaluated on train/validation/test splits")
    print("  5. Visualize the learning process")
    print()
    print("=" * 70)
    print()

    # Create environment and learner
    env = AdditionEnvironment(max_number=MAX_NUMBER)
    learner = AdditionLearner(env)

    # Train
    print("=" * 70)
    print("TRAINING ADDITION LEARNER (WITH VALIDATION)")
    print("=" * 70)

    results = learner.train(
        num_episodes=NUM_TRAINING_EPISODES,
        validation_size=VALIDATION_SET_SIZE,
        eval_frequency=EVAL_FREQUENCY,
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total problems attempted: {learner.scorer.get_statistics()['total_attempts']}")
    print(f"Correct answers: {learner.scorer.get_statistics()['correct_attempts']}")
    print(f"Final TRAINING accuracy: {results['final_train_accuracy']*100:.2f}%")
    print(f"Final TRAINING avg error: {results['final_train_error']:.2f}")
    print(f"Final VALIDATION accuracy: {results['final_val_accuracy']*100:.2f}%")
    print(f"Final VALIDATION avg error: {results['final_val_error']:.2f}")
    print(f"Knowledge base size: {results['knowledge_base_size']}")

    # Visualize
    visualize_addition_learning(results)

    # Test
    print("\n" + "=" * 70)
    print("TESTING LEARNED MODEL")
    print("=" * 70)

    test_results = learner.test(num_problems=TEST_SET_SIZE)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  - Model learned from trial and error")
    print("  - Accuracy improved over time")
    print("  - Proper train/validation/test split")
    print("  - Knowledge base grew with experience")
    print("  - Visualizations show clear learning dynamics")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for running examples."""
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = '--all'

    if mode in ['--basic', '--all']:
        run_basic_demos()

    if mode in ['--viz', '--visualization', '--all']:
        print("\n\n")
        run_visualization_demos()

    if mode in ['--addition', '--learning', '--all']:
        print("\n\n")
        run_addition_learning_demo()

    if mode not in ['--basic', '--viz', '--visualization', '--addition', '--learning', '--all']:
        print("Usage: python examples.py [--basic|--viz|--addition|--all]")
        print()
        print("Options:")
        print("  --basic      Run basic reasoning demos only")
        print("  --viz        Run visualization demos only")
        print("  --addition   Run addition learning demo only")
        print("  --all        Run all demos (default)")


if __name__ == "__main__":
    main()
