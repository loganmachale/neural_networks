"""
Visualization tools for the Logic Model (LM) Prototype

This module provides functions to track and visualize the model's performance
over time, including loss, scores, confidence levels, and action distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
from logic_model import LogicModel, InternalState, Goal, Experience


class MetricsTracker:
    """
    Tracks metrics during the reasoning process for visualization.
    """

    def __init__(self):
        self.epochs: List[int] = []
        self.scores: List[float] = []
        self.losses: List[float] = []
        self.confidences: List[float] = []
        self.action_counts: Dict[str, int] = {}
        self.state_sizes: List[int] = []

    def record_step(self, epoch: int, score: float, loss: float,
                   avg_confidence: float, action_type: str, state_size: int):
        """
        Record metrics for a single step.

        Args:
            epoch: Current iteration/epoch number
            score: Current score
            loss: Current loss (can be negative of score or separate metric)
            avg_confidence: Average confidence of all facts
            action_type: Type of action taken
            state_size: Number of facts in current state
        """
        self.epochs.append(epoch)
        self.scores.append(score)
        self.losses.append(loss)
        self.confidences.append(avg_confidence)
        self.state_sizes.append(state_size)

        # Track action distribution
        if action_type not in self.action_counts:
            self.action_counts[action_type] = 0
        self.action_counts[action_type] += 1

    def compute_avg_confidence(self, state: InternalState) -> float:
        """Compute average confidence across all facts in a state."""
        if not state.facts:
            return 0.0
        total = sum(conf for _, conf in state.facts.values())
        return total / len(state.facts)


class LogicModelWithTracking(LogicModel):
    """
    Extended LogicModel that tracks metrics for visualization.
    """

    def __init__(self, initial_state: InternalState, goal: Goal, search_depth: int = 3):
        super().__init__(initial_state, goal, search_depth)
        self.metrics = MetricsTracker()

    def think_step(self, verbose: bool = False) -> bool:
        """
        Execute one step with metric tracking.
        """
        # Get current metrics before step
        current_score = self.scoring_module.score(self.current_state, self.goal)
        avg_conf = self.metrics.compute_avg_confidence(self.current_state)
        state_size = len(self.current_state.facts)

        # Execute the step
        result = super().think_step(verbose)

        if result and self.history:
            # Get the action that was just taken
            _, last_action, new_score = self.history[-1]
            action_type = type(last_action).__name__

            # Loss is negative score (we want to minimize loss = maximize score)
            loss = -new_score

            # Record metrics
            self.metrics.record_step(
                epoch=self.iteration_count,
                score=new_score,
                loss=loss,
                avg_confidence=avg_conf,
                action_type=action_type,
                state_size=state_size
            )

        return result


def plot_loss_vs_epochs(metrics: MetricsTracker, title: str = "Loss vs Epochs",
                       save_path: Optional[str] = None):
    """
    Plot loss over epochs.

    Args:
        metrics: MetricsTracker instance with recorded data
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.epochs, metrics.losses, 'b-', linewidth=2, marker='o', markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")

    plt.show()


def plot_score_vs_epochs(metrics: MetricsTracker, title: str = "Score vs Epochs",
                        save_path: Optional[str] = None):
    """
    Plot score over epochs.

    Args:
        metrics: MetricsTracker instance with recorded data
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.epochs, metrics.scores, 'g-', linewidth=2, marker='s', markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved score plot to {save_path}")

    plt.show()


def plot_combined_metrics(metrics: MetricsTracker, title: str = "Training Metrics",
                         save_path: Optional[str] = None):
    """
    Plot loss and score on the same figure with dual y-axes.

    Args:
        metrics: MetricsTracker instance with recorded data
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot score on left y-axis
    color = 'tab:green'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Score', color=color, fontsize=12)
    ax1.plot(metrics.epochs, metrics.scores, color=color, linewidth=2,
             marker='s', markersize=6, label='Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot loss on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color, fontsize=12)
    ax2.plot(metrics.epochs, metrics.losses, color=color, linewidth=2,
             marker='o', markersize=6, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined metrics plot to {save_path}")

    plt.show()


def plot_confidence_vs_epochs(metrics: MetricsTracker,
                              title: str = "Average Confidence vs Epochs",
                              save_path: Optional[str] = None):
    """
    Plot average confidence over epochs.

    Args:
        metrics: MetricsTracker instance with recorded data
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.epochs, metrics.confidences, 'r-', linewidth=2,
             marker='^', markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confidence plot to {save_path}")

    plt.show()


def plot_action_distribution(metrics: MetricsTracker,
                             title: str = "Action Distribution",
                             save_path: Optional[str] = None):
    """
    Plot a bar chart showing the distribution of actions taken.

    Args:
        metrics: MetricsTracker instance with recorded data
        title: Plot title
        save_path: Optional path to save the figure
    """
    actions = list(metrics.action_counts.keys())
    counts = list(metrics.action_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(actions, counts, color='skyblue', edgecolor='navy', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Action Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved action distribution plot to {save_path}")

    plt.show()


def plot_all_metrics(metrics: MetricsTracker, title_prefix: str = "",
                    save_dir: Optional[str] = None):
    """
    Generate all available plots for the metrics.

    Args:
        metrics: MetricsTracker instance with recorded data
        title_prefix: Prefix for all plot titles
        save_dir: Optional directory to save all figures
    """
    if not metrics.epochs:
        print("No metrics to plot!")
        return

    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(16, 10))

    # 1. Score vs Epochs
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(metrics.epochs, metrics.scores, 'g-', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{title_prefix}Score vs Epochs')
    ax1.grid(True, alpha=0.3)

    # 2. Loss vs Epochs
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(metrics.epochs, metrics.losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{title_prefix}Loss vs Epochs')
    ax2.grid(True, alpha=0.3)

    # 3. Confidence vs Epochs
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(metrics.epochs, metrics.confidences, 'r-', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Avg Confidence')
    ax3.set_title(f'{title_prefix}Confidence vs Epochs')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # 4. State Size vs Epochs
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(metrics.epochs, metrics.state_sizes, 'm-', linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Number of Facts')
    ax4.set_title(f'{title_prefix}State Size vs Epochs')
    ax4.grid(True, alpha=0.3)

    # 5. Action Distribution
    ax5 = plt.subplot(2, 3, 5)
    actions = list(metrics.action_counts.keys())
    counts = list(metrics.action_counts.values())
    bars = ax5.bar(actions, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    ax5.set_xlabel('Action Type')
    ax5.set_ylabel('Count')
    ax5.set_title(f'{title_prefix}Action Distribution')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Score improvement rate
    ax6 = plt.subplot(2, 3, 6)
    if len(metrics.scores) > 1:
        score_diffs = np.diff(metrics.scores)
        ax6.plot(metrics.epochs[1:], score_diffs, 'c-', linewidth=2, marker='x', markersize=4)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Score Change')
        ax6.set_title(f'{title_prefix}Score Improvement Rate')
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = f"{save_dir}/dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dashboard to {save_path}")

    plt.show()


def create_training_summary(metrics: MetricsTracker) -> str:
    """
    Create a text summary of the training metrics.

    Args:
        metrics: MetricsTracker instance with recorded data

    Returns:
        Formatted summary string
    """
    if not metrics.epochs:
        return "No training data available."

    summary = []
    summary.append("=" * 60)
    summary.append("TRAINING SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Total Epochs: {len(metrics.epochs)}")
    summary.append(f"")
    summary.append(f"Score:")
    summary.append(f"  Initial: {metrics.scores[0]:.4f}")
    summary.append(f"  Final:   {metrics.scores[-1]:.4f}")
    summary.append(f"  Best:    {max(metrics.scores):.4f}")
    summary.append(f"  Change:  {metrics.scores[-1] - metrics.scores[0]:+.4f}")
    summary.append(f"")
    summary.append(f"Loss:")
    summary.append(f"  Initial: {metrics.losses[0]:.4f}")
    summary.append(f"  Final:   {metrics.losses[-1]:.4f}")
    summary.append(f"  Best:    {min(metrics.losses):.4f}")
    summary.append(f"  Change:  {metrics.losses[-1] - metrics.losses[0]:+.4f}")
    summary.append(f"")
    summary.append(f"Confidence:")
    summary.append(f"  Initial: {metrics.confidences[0]:.4f}")
    summary.append(f"  Final:   {metrics.confidences[-1]:.4f}")
    summary.append(f"  Average: {np.mean(metrics.confidences):.4f}")
    summary.append(f"")
    summary.append(f"Actions Taken:")
    for action, count in sorted(metrics.action_counts.items(), key=lambda x: -x[1]):
        summary.append(f"  {action}: {count}")
    summary.append("=" * 60)

    return "\n".join(summary)
