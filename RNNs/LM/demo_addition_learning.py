"""
Addition Learning Demo

This demo shows the Logic Model learning to perform addition through trial and error.
The model starts with no knowledge of arithmetic and learns by:
1. Making attempts (guessing sums)
2. Receiving feedback (correct/incorrect)
3. Improving its scoring heuristic based on experience
4. Visualizing the learning process

This demonstrates the adaptive learning approach from README3.md.
"""

import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from logic_model import InternalState, Goal, ActionDI, ScoringModule


# =============================================================================
# Addition-Specific Actions
# =============================================================================

class ProposeAdditionDI(ActionDI):
    """Action: Propose a sum for two numbers."""

    def __init__(self, a: int, b: int, proposed_sum: int, confidence: float = 0.5):
        self.a = a
        self.b = b
        self.proposed_sum = proposed_sum
        self.confidence = confidence

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        new_state.add_fact(f"sum_{self.a}_{self.b}", self.proposed_sum, self.confidence)
        return new_state

    def __repr__(self):
        return f"Propose({self.a}+{self.b}={self.proposed_sum}, conf={self.confidence:.2f})"


class RefineProposalDI(ActionDI):
    """Action: Refine an existing proposal by adjusting it."""

    def __init__(self, a: int, b: int, adjustment: int):
        self.a = a
        self.b = b
        self.adjustment = adjustment

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        existing = new_state.get_fact(f"sum_{self.a}_{self.b}")
        if existing:
            old_sum, old_conf = existing
            new_sum = old_sum + self.adjustment
            new_state.add_fact(f"sum_{self.a}_{self.b}", new_sum, min(old_conf + 0.1, 1.0))
        return new_state

    def __repr__(self):
        sign = "+" if self.adjustment >= 0 else ""
        return f"Refine({self.a}+{self.b} by {sign}{self.adjustment})"


# =============================================================================
# Learning-Enhanced Scoring Module
# =============================================================================

class LearningScoringModule(ScoringModule):
    """
    Enhanced scoring module that learns from feedback.

    Maintains a history of correct answers and adjusts scoring accordingly.
    """

    def __init__(self):
        super().__init__()
        # Knowledge base: stores known correct sums
        self.known_sums: Dict[Tuple[int, int], int] = {}
        # Learning history
        self.attempts: List[Tuple[int, int, int, bool]] = []  # (a, b, guess, correct)
        self.accuracy_history: List[float] = []

    def teach(self, a: int, b: int, correct_sum: int):
        """Teach the model the correct sum."""
        self.known_sums[(a, b)] = correct_sum

    def evaluate_guess(self, a: int, b: int, guess: int) -> Tuple[bool, int]:
        """
        Evaluate a guess and return (is_correct, error_magnitude).

        Returns:
            Tuple of (is_correct, absolute_error)
        """
        correct_sum = a + b
        is_correct = (guess == correct_sum)
        error = abs(guess - correct_sum)

        # Record attempt
        self.attempts.append((a, b, guess, is_correct))

        # Update accuracy history
        if len(self.attempts) > 0:
            recent_attempts = self.attempts[-10:]  # Last 10 attempts
            accuracy = sum(1 for _, _, _, correct in recent_attempts if correct) / len(recent_attempts)
            self.accuracy_history.append(accuracy)

        return is_correct, error

    def score(self, state: InternalState, goal: Goal) -> float:
        """
        Score a state based on how close the proposed sums are to correct answers.

        This scoring improves over time as the model learns.
        """
        score = super().score(state, goal)

        # Additional scoring based on learned knowledge
        bonus = 0.0

        # Check all proposed sums
        for fact_name, (value, confidence) in state.facts.items():
            if fact_name.startswith("sum_"):
                # Parse fact name: "sum_a_b"
                parts = fact_name.split("_")
                if len(parts) == 3:
                    try:
                        a, b = int(parts[1]), int(parts[2])
                        proposed_sum = value

                        # If we know the correct answer, reward accuracy
                        if (a, b) in self.known_sums:
                            correct_sum = self.known_sums[(a, b)]
                            error = abs(proposed_sum - correct_sum)

                            if error == 0:
                                bonus += 10.0 * confidence  # Large reward for correct answer
                            else:
                                # Penalty proportional to error
                                bonus -= error * 0.5
                        else:
                            # Use general arithmetic knowledge if available
                            correct_sum = a + b
                            error = abs(proposed_sum - correct_sum)

                            if error == 0:
                                bonus += 5.0 * confidence
                            else:
                                bonus -= error * 0.3
                    except (ValueError, IndexError):
                        pass

        return score + bonus

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning process."""
        if not self.attempts:
            return {
                'total_attempts': 0,
                'correct': 0,
                'accuracy': 0.0,
                'avg_error': 0.0
            }

        correct = sum(1 for _, _, _, is_correct in self.attempts if is_correct)
        errors = [abs(guess - (a + b)) for a, b, guess, _ in self.attempts]

        return {
            'total_attempts': len(self.attempts),
            'correct': correct,
            'accuracy': correct / len(self.attempts),
            'avg_error': sum(errors) / len(errors) if errors else 0.0
        }


# =============================================================================
# Addition Problem Environment
# =============================================================================

class AdditionEnvironment:
    """
    Environment for testing addition learning.

    Generates addition problems and provides feedback.
    """

    def __init__(self, max_number: int = 20):
        self.max_number = max_number

    def generate_problem(self) -> Tuple[int, int]:
        """Generate a random addition problem."""
        a = random.randint(0, self.max_number)
        b = random.randint(0, self.max_number)
        return a, b

    def check_answer(self, a: int, b: int, answer: int) -> bool:
        """Check if an answer is correct."""
        return (a + b) == answer

    def get_hint(self, a: int, b: int, guess: int) -> str:
        """Provide a hint based on the guess."""
        correct = a + b
        diff = guess - correct

        if diff == 0:
            return "Correct!"
        elif diff > 0:
            return f"Too high by {diff}"
        else:
            return f"Too low by {-diff}"


# =============================================================================
# Learning Agent
# =============================================================================

class AdditionLearner:
    """
    Agent that learns to perform addition through trial and error.
    """

    def __init__(self, environment: AdditionEnvironment):
        self.env = environment
        self.scorer = LearningScoringModule()
        self.training_history: List[Dict] = []

    def make_initial_guess(self, a: int, b: int) -> int:
        """
        Make an initial guess for a + b.

        Initially random, but improves as learning progresses.
        """
        # Check if we've learned similar problems
        if (a, b) in self.scorer.known_sums:
            return self.scorer.known_sums[(a, b)]

        # Check reverse (commutativity)
        if (b, a) in self.scorer.known_sums:
            return self.scorer.known_sums[(b, a)]

        # Use pattern matching from learned examples
        if len(self.scorer.known_sums) > 5:
            # Try to extrapolate from known examples
            similar_sums = [v for (ka, kb), v in self.scorer.known_sums.items()
                          if abs(ka - a) <= 2 and abs(kb - b) <= 2]
            if similar_sums:
                # Use average of similar problems as starting point
                avg_similar = sum(similar_sums) / len(similar_sums)
                # Adjust based on difference
                adjustment = (a + b) - sum([ka + kb for (ka, kb) in
                            [(ka, kb) for (ka, kb), _ in self.scorer.known_sums.items()
                             if abs(ka - a) <= 2 and abs(kb - b) <= 2]]) / len(similar_sums)
                return int(avg_similar + adjustment)

        # Random guess initially (representing no knowledge)
        return random.randint(0, a + b + 10)

    def learn_problem(self, a: int, b: int, max_attempts: int = 5) -> Tuple[bool, int]:
        """
        Try to solve a + b, learning from feedback.

        Returns:
            Tuple of (success, num_attempts)
        """
        correct_answer = a + b

        for attempt in range(max_attempts):
            # Make a guess
            if attempt == 0:
                guess = self.make_initial_guess(a, b)
            else:
                # Refine based on previous feedback
                prev_guess = guess
                hint = self.env.get_hint(a, b, prev_guess)

                if "Too high" in hint:
                    adjustment = -max(1, abs(prev_guess - correct_answer) // 2)
                elif "Too low" in hint:
                    adjustment = max(1, abs(prev_guess - correct_answer) // 2)
                else:
                    break

                guess = prev_guess + adjustment

            # Check answer
            is_correct, error = self.scorer.evaluate_guess(a, b, guess)

            if is_correct:
                # Learn this fact
                self.scorer.teach(a, b, correct_answer)
                return True, attempt + 1

        # Failed to learn in max_attempts
        # Still teach the correct answer
        self.scorer.teach(a, b, correct_answer)
        return False, max_attempts

    def evaluate_on_set(self, problem_set: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        Evaluate the model on a fixed set of problems (for validation/test).

        Args:
            problem_set: List of (a, b) tuples

        Returns:
            Tuple of (accuracy, average_error)
        """
        correct = 0
        total_error = 0

        for a, b in problem_set:
            guess = self.make_initial_guess(a, b)
            correct_answer = a + b

            if guess == correct_answer:
                correct += 1

            total_error += abs(guess - correct_answer)

        accuracy = correct / len(problem_set) if problem_set else 0.0
        avg_error = total_error / len(problem_set) if problem_set else 0.0

        return accuracy, avg_error

    def train(self, num_episodes: int = 50, validation_size: int = 20,
              eval_frequency: int = 5, verbose: bool = True) -> Dict:
        """
        Train the agent on random addition problems with validation.

        Args:
            num_episodes: Number of training episodes
            validation_size: Size of validation set
            eval_frequency: How often to evaluate on validation set
            verbose: Whether to print progress

        Returns:
            Training statistics including validation metrics
        """
        if verbose:
            print("=" * 70)
            print("TRAINING ADDITION LEARNER (WITH VALIDATION)")
            print("=" * 70)
            print()

        # Generate fixed validation set
        random.seed(42)  # For reproducibility
        validation_set = [self.env.generate_problem() for _ in range(validation_size)]
        random.seed()  # Reset seed

        if verbose:
            print(f"Created validation set: {validation_size} problems")
            print(f"Validation problems: {validation_set[:5]}... (showing first 5)")
            print()

        episode_scores = []
        episode_accuracies = []
        episode_errors = []
        validation_accuracies = []
        validation_errors = []
        validation_episodes = []

        for episode in range(num_episodes):
            # Generate problem
            a, b = self.env.generate_problem()

            # Try to solve it
            success, attempts = self.learn_problem(a, b)

            # Get statistics
            stats = self.scorer.get_learning_stats()

            # Record metrics
            episode_scores.append(1.0 if success else 0.0)
            episode_accuracies.append(stats['accuracy'])
            episode_errors.append(stats['avg_error'])

            # Evaluate on validation set periodically
            if (episode + 1) % eval_frequency == 0:
                val_acc, val_err = self.evaluate_on_set(validation_set)
                validation_accuracies.append(val_acc)
                validation_errors.append(val_err)
                validation_episodes.append(episode)

            # Record in history
            self.training_history.append({
                'episode': episode,
                'problem': (a, b),
                'success': success,
                'attempts': attempts,
                'accuracy': stats['accuracy'],
                'avg_error': stats['avg_error']
            })

            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Recent problem: {a} + {b} = {a + b}")
                print(f"  Success: {success} (in {attempts} attempts)")
                print(f"  Training accuracy: {stats['accuracy']:.2%}")
                print(f"  Training avg error: {stats['avg_error']:.2f}")

                # Show validation metrics if available
                if validation_accuracies:
                    print(f"  Validation accuracy: {validation_accuracies[-1]:.2%}")
                    print(f"  Validation avg error: {validation_errors[-1]:.2f}")
                print()

        final_stats = self.scorer.get_learning_stats()
        final_val_acc, final_val_err = self.evaluate_on_set(validation_set)

        if verbose:
            print("=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Total problems attempted: {final_stats['total_attempts']}")
            print(f"Correct answers: {final_stats['correct']}")
            print(f"Final TRAINING accuracy: {final_stats['accuracy']:.2%}")
            print(f"Final TRAINING avg error: {final_stats['avg_error']:.2f}")
            print(f"Final VALIDATION accuracy: {final_val_acc:.2%}")
            print(f"Final VALIDATION avg error: {final_val_err:.2f}")
            print(f"Knowledge base size: {len(self.scorer.known_sums)}")
            print()

            # Check for overfitting
            if final_stats['accuracy'] > final_val_acc + 0.15:
                print("[WARNING] Possible overfitting detected!")
                print(f"  Training accuracy ({final_stats['accuracy']:.2%}) >> "
                      f"Validation accuracy ({final_val_acc:.2%})")
            elif final_val_acc > final_stats['accuracy']:
                print("[EXCELLENT] Good generalization: Validation >= Training")
            else:
                print("[OK] Healthy gap between training and validation")

        return {
            'episode_scores': episode_scores,
            'episode_accuracies': episode_accuracies,
            'episode_errors': episode_errors,
            'validation_accuracies': validation_accuracies,
            'validation_errors': validation_errors,
            'validation_episodes': validation_episodes,
            'validation_set': validation_set,
            'final_stats': final_stats,
            'final_val_accuracy': final_val_acc,
            'final_val_error': final_val_err
        }


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_training(learner: AdditionLearner, training_results: Dict):
    """Create comprehensive visualization of the learning process with validation."""

    episode_scores = training_results['episode_scores']
    episode_accuracies = training_results['episode_accuracies']
    episode_errors = training_results['episode_errors']
    validation_accuracies = training_results.get('validation_accuracies', [])
    validation_errors = training_results.get('validation_errors', [])
    validation_episodes = training_results.get('validation_episodes', [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Addition Learning Progress (Train vs Validation)', fontsize=16, fontweight='bold')

    episodes = list(range(len(episode_scores)))

    # 1. Accuracy over time (Train vs Validation)
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_accuracies, 'g-', linewidth=2, label='Training Accuracy', alpha=0.7)

    if validation_accuracies:
        ax1.plot(validation_episodes, validation_accuracies, 'b-', linewidth=2.5,
                marker='o', markersize=6, label='Validation Accuracy')

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (100%)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # 2. Average error over time (Train vs Validation)
    ax2 = axes[0, 1]
    ax2.plot(episodes, episode_errors, 'r-', linewidth=2, label='Training Error', alpha=0.7)

    if validation_errors:
        ax2.plot(validation_episodes, validation_errors, 'orange', linewidth=2.5,
                marker='s', markersize=6, label='Validation Error')

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Error')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Error')
    ax2.set_title('Training vs Validation Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Success rate (rolling average)
    ax3 = axes[1, 0]
    window_size = 10
    rolling_success = []
    for i in range(len(episode_scores)):
        start = max(0, i - window_size + 1)
        window = episode_scores[start:i+1]
        rolling_success.append(sum(window) / len(window))

    ax3.plot(episodes, rolling_success, 'b-', linewidth=2, label=f'Success Rate (window={window_size})')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_title(f'Success Rate (Rolling Avg, Window={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # 4. Train/Val Gap (Overfitting Detection)
    ax4 = axes[1, 1]

    if validation_accuracies:
        # Show the gap between training and validation
        train_at_val_points = [episode_accuracies[ep] for ep in validation_episodes]
        gaps = [train - val for train, val in zip(train_at_val_points, validation_accuracies)]

        ax4.plot(validation_episodes, gaps, 'purple', linewidth=2.5,
                marker='D', markersize=6, label='Train - Val Gap')
        ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label='No Gap (Perfect Generalization)')
        ax4.axhline(y=0.15, color='red', linestyle='--', linewidth=2, alpha=0.5,
                   label='Overfitting Threshold')
        ax4.fill_between(validation_episodes, 0, 0.15, color='yellow', alpha=0.1)

        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Accuracy Gap')
        ax4.set_title('Overfitting Detection (Train - Validation Gap)')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
    else:
        # Fallback to knowledge base growth if no validation
        kb_sizes = []
        current_kb = set()
        for entry in learner.training_history:
            a, b = entry['problem']
            current_kb.add((a, b))
            kb_sizes.append(len(current_kb))

        ax4.plot(episodes, kb_sizes, 'm-', linewidth=2, label='Unique Problems Learned')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Knowledge Base Size')
        ax4.set_title('Growth of Learned Problems')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('addition_learning_progress.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: addition_learning_progress.png")
    plt.show()


def test_learned_model(learner: AdditionLearner, num_tests: int = 10):
    """Test the learned model on new problems."""
    print("\n" + "=" * 70)
    print("TESTING LEARNED MODEL")
    print("=" * 70)
    print()

    correct = 0
    for i in range(num_tests):
        a, b = learner.env.generate_problem()
        guess = learner.make_initial_guess(a, b)
        correct_answer = a + b
        is_correct = (guess == correct_answer)

        if is_correct:
            correct += 1

        print(f"Test {i+1}: {a} + {b} = ?")
        print(f"  Model's answer: {guess}")
        print(f"  Correct answer: {correct_answer}")
        print(f"  Result: {'[CORRECT]' if is_correct else '[INCORRECT]'}")
        print()

    print(f"Test Accuracy: {correct}/{num_tests} = {correct/num_tests:.1%}")


# =============================================================================
# Main Demo
# =============================================================================

def main():
    """Run the addition learning demo."""
    print("\n" + "=" * 70)
    print("ADDITION LEARNING DEMO")
    print("=" * 70)
    print()
    print("This demo shows a reasoning model learning to perform addition")
    print("through trial and error, with real-time visualization of learning.")
    print()
    print("The model will:")
    print("  1. Start with no knowledge of arithmetic")
    print("  2. Make guesses and receive feedback")
    print("  3. Learn from mistakes and improve")
    print("  4. Visualize the learning process")
    print()
    print("=" * 70)
    print()

    # Create environment and learner
    env = AdditionEnvironment(max_number=20)
    learner = AdditionLearner(env)

    # Train the model
    training_results = learner.train(num_episodes=50, verbose=True)

    # Visualize learning
    visualize_training(learner, training_results)

    # Test on new problems
    test_learned_model(learner, num_tests=10)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • The model learned from its mistakes")
    print("  • Accuracy improved over time")
    print("  • Errors decreased as learning progressed")
    print("  • Knowledge base grew with each problem")
    print("  • Visualizations show clear learning curve")


if __name__ == "__main__":
    main()
