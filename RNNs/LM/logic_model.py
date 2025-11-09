"""
Logic Model (LM) Prototype
Based on g1-g3 (project, assess, iterate) and p1-p4 (simulate, score, search, store) framework.

This model uses tree search to find optimal chains of internal reasoning actions.
"""

import copy
import math
from typing import Dict, Tuple, List, Optional, Callable, Any
from abc import ABC, abstractmethod


# =============================================================================
# Phase 1: Core Data Structures
# =============================================================================

class InternalState:
    """
    Represents the AI's internal belief state.

    The state contains:
    - facts: Explicit beliefs with values and confidence levels
    - rules: Logical rules that can derive new facts from existing ones
    """

    def __init__(self):
        # Symbolic store: {fact_name: (value, confidence)}
        # Example: {"sky_is_blue": (True, 0.99)}
        self.facts: Dict[str, Tuple[Any, float]] = {}

        # Logical rules: Functions that can add new facts based on existing ones
        # Example: lambda facts: ("is_mortal", "socrates") if facts.get(("is_man", "socrates")) else None
        self.rules: List[Callable] = []

    def copy(self):
        """Create a deep copy of the state for simulation purposes."""
        new_state = InternalState()
        new_state.facts = copy.deepcopy(self.facts)
        new_state.rules = copy.copy(self.rules)  # Rules are functions, shallow copy is fine
        return new_state

    def add_fact(self, fact_name: str, value: Any, confidence: float):
        """Add or update a fact in the belief state."""
        self.facts[fact_name] = (value, confidence)

    def get_fact(self, fact_name: str) -> Optional[Tuple[Any, float]]:
        """Retrieve a fact from the belief state."""
        return self.facts.get(fact_name)

    def add_rule(self, rule: Callable):
        """Add a logical rule to the state."""
        self.rules.append(rule)

    def apply_rules(self):
        """Apply all logical rules to derive new facts."""
        for rule in self.rules:
            result = rule(self.facts)
            if result:
                fact_name, value, confidence = result
                self.add_fact(fact_name, value, confidence)

    def __hash__(self):
        """
        Generate a hash for this state for use in memoization/caching.

        Idea from README2.md: Enable efficient state caching in search.
        """
        # Create a hashable representation of the facts
        # Sort to ensure consistent hashing
        fact_items = tuple(sorted(self.facts.items()))
        return hash(fact_items)

    def __eq__(self, other):
        """Check equality based on facts (for caching purposes)."""
        if not isinstance(other, InternalState):
            return False
        return self.facts == other.facts

    def achieves_goal(self, goal: 'Goal', threshold: float = 0.9) -> bool:
        """
        Check if this state achieves the specified goal.

        Idea from README2.md: Explicit goal checking for early termination.

        Args:
            goal: The goal to check against
            threshold: Confidence threshold for considering a goal fact achieved

        Returns:
            True if all goal facts are present with sufficient confidence
        """
        for goal_fact_name, (goal_value, goal_confidence) in goal.goal_facts.items():
            state_fact = self.get_fact(goal_fact_name)

            if not state_fact:
                return False  # Goal fact not present

            state_value, state_confidence = state_fact

            if state_value != goal_value:
                return False  # Wrong value

            if state_confidence < goal_confidence * threshold:
                return False  # Insufficient confidence

        return True

    def __repr__(self):
        facts_str = "\n  ".join([f"{k}: {v}" for k, v in self.facts.items()])
        return f"InternalState(\n  {facts_str}\n)"


class ActionDI(ABC):
    """
    Base class for "internal actions" (di).

    These are the "verbs" of the system - actions that modify the belief state.
    All actions must implement apply() which returns a NEW state (no in-place mutation).
    """

    @abstractmethod
    def apply(self, state: InternalState) -> InternalState:
        """
        Apply this action to a state and return a new state.

        Args:
            state: The current InternalState

        Returns:
            A new InternalState with the action applied
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass


class AddFactDI(ActionDI):
    """Action to add a new fact to the belief state."""

    def __init__(self, fact_name: str, value: Any, confidence: float):
        self.fact_name = fact_name
        self.value = value
        self.confidence = confidence

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        new_state.add_fact(self.fact_name, self.value, self.confidence)
        return new_state

    def __repr__(self):
        return f"AddFact({self.fact_name}={self.value}, conf={self.confidence})"


class UpdateFactConfidenceDI(ActionDI):
    """Action to update the confidence level of an existing fact."""

    def __init__(self, fact_name: str, new_confidence: float):
        self.fact_name = fact_name
        self.new_confidence = new_confidence

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        existing = new_state.get_fact(self.fact_name)
        if existing:
            value, _ = existing
            new_state.add_fact(self.fact_name, value, self.new_confidence)
        return new_state

    def __repr__(self):
        return f"UpdateConfidence({self.fact_name}, conf={self.new_confidence})"


class RefuteFactDI(ActionDI):
    """Action to add a contradiction (refute a fact)."""

    def __init__(self, fact_name: str, confidence: float = 0.9):
        self.fact_name = fact_name
        self.confidence = confidence

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        existing = new_state.get_fact(self.fact_name)
        if existing:
            value, _ = existing
            # Add the negation
            new_state.add_fact(f"NOT_{self.fact_name}", not value if isinstance(value, bool) else None, self.confidence)
        return new_state

    def __repr__(self):
        return f"Refute({self.fact_name})"


class ApplyRuleDI(ActionDI):
    """Action to apply a specific logical rule."""

    def __init__(self, rule_index: int):
        self.rule_index = rule_index

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        if 0 <= self.rule_index < len(new_state.rules):
            rule = new_state.rules[self.rule_index]
            result = rule(new_state.facts)
            if result:
                fact_name, value, confidence = result
                new_state.add_fact(fact_name, value, confidence)
        return new_state

    def __repr__(self):
        return f"ApplyRule({self.rule_index})"


class AddContradictionDI(ActionDI):
    """
    Adversarial action: Add a contradicting fact to challenge beliefs.
    Used by the "min" player in the search algorithm.
    """

    def __init__(self, fact_name: str, contradicting_value: Any, confidence: float = 0.5):
        self.fact_name = fact_name
        self.contradicting_value = contradicting_value
        self.confidence = confidence

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        new_state.add_fact(f"CONTRA_{self.fact_name}", self.contradicting_value, self.confidence)
        return new_state

    def __repr__(self):
        return f"AddContradiction({self.fact_name})"


class LowerConfidenceDI(ActionDI):
    """
    Adversarial action: Lower the confidence of a fact.
    Used by the "min" player in the search algorithm.
    """

    def __init__(self, fact_name: str, reduction: float = 0.2):
        self.fact_name = fact_name
        self.reduction = reduction

    def apply(self, state: InternalState) -> InternalState:
        new_state = state.copy()
        existing = new_state.get_fact(self.fact_name)
        if existing:
            value, confidence = existing
            new_confidence = max(0.0, confidence - self.reduction)
            new_state.add_fact(self.fact_name, value, new_confidence)
        return new_state

    def __repr__(self):
        return f"LowerConfidence({self.fact_name})"


class Goal:
    """
    Represents a target state that the LM is trying to achieve.

    The goal contains facts that should be present with high confidence.
    """

    def __init__(self, goal_facts: Dict[str, Tuple[Any, float]]):
        """
        Args:
            goal_facts: Dictionary of {fact_name: (desired_value, desired_confidence)}
        """
        self.goal_facts = goal_facts

    def __repr__(self):
        facts_str = ", ".join([f"{k}={v}" for k, v in self.goal_facts.items()])
        return f"Goal({facts_str})"


# =============================================================================
# Phase 2: Simulation & Scoring (p1 & p2)
# =============================================================================

class Simulator:
    """
    Simulator for applying chains of actions to states (p1).

    This allows the model to "think ahead" by simulating sequences of actions.
    """

    @staticmethod
    def simulate(initial_state: InternalState, action_chain: List[ActionDI]) -> InternalState:
        """
        Apply a chain of actions to an initial state.

        Args:
            initial_state: The starting state
            action_chain: List of actions to apply in sequence

        Returns:
            The final state after all actions are applied
        """
        current_state = initial_state.copy()
        for action in action_chain:
            current_state = action.apply(current_state)
        return current_state


class ScoringModule:
    """
    Heuristic evaluation function for belief states (p2).

    This is the core evaluation function that determines how "good" a state is.
    The score guides the search algorithm toward better reasoning.
    """

    def __init__(self,
                 consistency_weight: float = 10.0,
                 goal_alignment_weight: float = 5.0,
                 parsimony_weight: float = -0.1,
                 confidence_weight: float = 1.0):
        """
        Args:
            consistency_weight: Weight for consistency score (penalize contradictions)
            goal_alignment_weight: Weight for goal alignment
            parsimony_weight: Weight for parsimony (simpler is better)
            confidence_weight: Weight for overall confidence
        """
        self.consistency_weight = consistency_weight
        self.goal_alignment_weight = goal_alignment_weight
        self.parsimony_weight = parsimony_weight
        self.confidence_weight = confidence_weight

    def score(self, state: InternalState, goal: Goal) -> float:
        """
        Evaluate a belief state and return a desirability score.

        Args:
            state: The state to evaluate
            goal: The goal we're trying to achieve

        Returns:
            A score where higher is better
        """
        total_score = 0.0
        total_score += self.consistency_weight * self.score_consistency(state)
        total_score += self.goal_alignment_weight * self.score_goal_alignment(state, goal)
        total_score += self.parsimony_weight * self.score_parsimony(state)
        total_score += self.confidence_weight * self.score_confidence(state)
        return total_score

    def score_consistency(self, state: InternalState) -> float:
        """
        Check for contradictions in the belief state.

        Returns a large negative penalty if contradictions are found.
        """
        score = 0.0
        fact_names = set()

        # Collect base fact names
        for fact_name in state.facts.keys():
            if fact_name.startswith("NOT_"):
                base_name = fact_name[4:]
                fact_names.add(base_name)
            elif fact_name.startswith("CONTRA_"):
                base_name = fact_name[7:]
                fact_names.add(base_name)
            else:
                fact_names.add(fact_name)

        # Check for contradictions
        for base_name in fact_names:
            base_fact = state.get_fact(base_name)
            not_fact = state.get_fact(f"NOT_{base_name}")
            contra_fact = state.get_fact(f"CONTRA_{base_name}")

            if base_fact and not_fact:
                # Both fact and its negation exist
                _, base_conf = base_fact
                _, not_conf = not_fact
                # Penalty is proportional to the confidence of both
                penalty = -(base_conf * not_conf)
                score += penalty

            if base_fact and contra_fact:
                # Fact and a contradiction exist
                base_val, base_conf = base_fact
                contra_val, contra_conf = contra_fact
                if base_val != contra_val:
                    penalty = -(base_conf * contra_conf)
                    score += penalty

        return score

    def score_goal_alignment(self, state: InternalState, goal: Goal) -> float:
        """
        Measure how well the state aligns with the goal.

        Returns a positive score based on how many goal facts are achieved.
        """
        score = 0.0

        for goal_fact_name, (goal_value, goal_confidence) in goal.goal_facts.items():
            state_fact = state.get_fact(goal_fact_name)
            if state_fact:
                state_value, state_confidence = state_fact
                if state_value == goal_value:
                    # Reward based on how close we are to the desired confidence
                    confidence_match = min(state_confidence / goal_confidence, 1.0)
                    score += confidence_match
                else:
                    # Wrong value, penalty
                    score -= 0.5
            else:
                # Fact not present, small penalty
                score -= 0.2

        return score

    def score_parsimony(self, state: InternalState) -> float:
        """
        Prefer simpler belief states (Occam's razor).

        Returns a score based on the number of facts (more facts = lower score).
        """
        return -len(state.facts)

    def score_confidence(self, state: InternalState) -> float:
        """
        Prefer states with high-confidence beliefs.

        Returns the average confidence across all facts.
        """
        if not state.facts:
            return 0.0

        total_confidence = sum(conf for _, conf in state.facts.values())
        return total_confidence / len(state.facts)


# =============================================================================
# Phase 3: Search Algorithm (p3) - The "Chess Engine"
# =============================================================================

class SearchModule:
    """
    Tree search algorithm using Minimax with Alpha-Beta pruning (p3).

    This is like a chess engine, but for reasoning. It searches through
    possible chains of internal actions to find the best reasoning path.

    The "min" player represents Uncertainty/Environment that challenges beliefs.
    The "max" player represents the LM trying to improve its belief state.

    Idea from README2.md: Uses memoization/transposition table for efficiency.
    """

    def __init__(self, scoring_module: ScoringModule, use_memoization: bool = True):
        self.scoring_module = scoring_module
        self.use_memoization = use_memoization
        # Memoization cache: maps (state_hash, depth, is_max_player) -> (score, best_action)
        self.memo_cache: Dict[Tuple[int, int, bool], Tuple[float, Optional[ActionDI]]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cache(self):
        """Clear the memoization cache."""
        self.memo_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'cache_size': len(self.memo_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

    def get_possible_actions(self, state: InternalState, is_max_player: bool) -> List[ActionDI]:
        """
        Generate possible actions for the current player.

        Args:
            state: Current belief state
            is_max_player: True if LM's turn, False if Uncertainty's turn

        Returns:
            List of possible actions
        """
        actions = []

        if is_max_player:
            # Max player (LM) performs constructive actions

            # 1. PRIORITY: Apply rules if any exist (put these first!)
            for i in range(len(state.rules)):
                actions.append(ApplyRuleDI(i))

            # 2. Update confidence of existing facts
            for fact_name in state.facts.keys():
                if not fact_name.startswith("NOT_") and not fact_name.startswith("CONTRA_"):
                    actions.append(UpdateFactConfidenceDI(fact_name, 0.95))

        else:
            # Min player (Uncertainty) performs adversarial actions
            # 1. Lower confidence of existing facts
            for fact_name in state.facts.keys():
                if not fact_name.startswith("NOT_") and not fact_name.startswith("CONTRA_"):
                    actions.append(LowerConfidenceDI(fact_name, 0.1))

            # 2. Add contradictions
            for fact_name, (value, _) in state.facts.items():
                if not fact_name.startswith("NOT_") and not fact_name.startswith("CONTRA_"):
                    if isinstance(value, bool):
                        actions.append(AddContradictionDI(fact_name, not value, 0.3))

        # Limit action space to prevent combinatorial explosion
        return actions[:10]  # Take at most 10 actions

    def is_terminal_node(self, state: InternalState, goal: Goal, depth: int) -> bool:
        """
        Check if we've reached a terminal node in the search.

        Idea from README2.md: Check goal achievement for early termination.

        Args:
            state: Current state
            goal: The goal
            depth: Current search depth

        Returns:
            True if terminal, False otherwise
        """
        # Terminal if depth is 0
        if depth == 0:
            return True

        # Terminal if goal is achieved
        if state.achieves_goal(goal):
            return True

        return False

    def alpha_beta_search(self,
                         state: InternalState,
                         goal: Goal,
                         depth: int,
                         alpha: float,
                         beta: float,
                         is_max_player: bool) -> Tuple[float, Optional[ActionDI]]:
        """
        Minimax search with alpha-beta pruning.

        Idea from README2.md: Uses memoization to avoid recomputing known states.

        Args:
            state: Current belief state
            goal: The goal to achieve
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_max_player: True if LM's turn, False if Uncertainty's turn

        Returns:
            Tuple of (best_score, best_action)
        """
        # Check memoization cache
        if self.use_memoization:
            cache_key = (hash(state), depth, is_max_player)
            if cache_key in self.memo_cache:
                self.cache_hits += 1
                return self.memo_cache[cache_key]
            self.cache_misses += 1

        # Base case: terminal node
        if self.is_terminal_node(state, goal, depth):
            score = self.scoring_module.score(state, goal)
            return score, None

        possible_actions = self.get_possible_actions(state, is_max_player)

        # If no actions available, return current score
        if not possible_actions:
            return self.scoring_module.score(state, goal), None

        best_action = None

        if is_max_player:
            # Maximizing player
            max_eval = float('-inf')
            for action in possible_actions:
                new_state = action.apply(state)
                eval_score, _ = self.alpha_beta_search(new_state, goal, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff

            result = (max_eval, best_action)
        else:
            # Minimizing player
            min_eval = float('inf')
            for action in possible_actions:
                new_state = action.apply(state)
                eval_score, _ = self.alpha_beta_search(new_state, goal, depth - 1, alpha, beta, True)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff

            result = (min_eval, best_action)

        # Store in memoization cache
        if self.use_memoization:
            cache_key = (hash(state), depth, is_max_player)
            self.memo_cache[cache_key] = result

        return result

    def find_best_action(self, state: InternalState, goal: Goal, search_depth: int = 3) -> ActionDI:
        """
        Find the best action to take from the current state.

        Args:
            state: Current belief state
            goal: The goal to achieve
            search_depth: How many moves to look ahead

        Returns:
            The best action to take
        """
        _, best_action = self.alpha_beta_search(
            state, goal, search_depth,
            alpha=float('-inf'),
            beta=float('inf'),
            is_max_player=True
        )
        return best_action


# =============================================================================
# Monte Carlo Tree Search (MCTS)
# =============================================================================

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.

    Stores state and search statistics for MCTS algorithm.
    """

    def __init__(self, state: InternalState, parent: Optional['MCTSNode'] = None,
                 action: Optional[ActionDI] = None, prior_prob: float = 1.0):
        """
        Initialize MCTS node.

        Args:
            state: Belief state at this node
            parent: Parent node in tree
            action: Action that led to this node
            prior_prob: Prior probability from policy (default uniform)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, 'MCTSNode'] = {}

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob

    def is_leaf(self) -> bool:
        """Check if node has no children."""
        return len(self.children) == 0

    def get_value(self) -> float:
        """Calculate average value from visits."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Calculate Upper Confidence Bound score for node selection.

        Balances exploitation (high value) with exploration (low visits).

        Args:
            c_puct: Exploration constant

        Returns:
            UCB score
        """
        if self.visit_count == 0:
            return float('inf')

        # UCB1 formula
        exploitation = self.get_value()
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        return exploitation + exploration


class MCTSSearch:
    """
    Monte Carlo Tree Search for reasoning.

    Alternative to Alpha-Beta that uses simulation and statistical sampling.
    """

    def __init__(self, scoring_module: ScoringModule, num_simulations: int = 100,
                 c_puct: float = 1.0):
        """
        Initialize MCTS search.

        Args:
            scoring_module: Scoring function for state evaluation
            num_simulations: Number of MCTS rollouts
            c_puct: Exploration constant for UCB
        """
        self.scoring_module = scoring_module
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state: InternalState, goal: Goal,
               available_actions: List[ActionDI]) -> ActionDI:
        """
        Run MCTS to find best action.

        Args:
            root_state: Starting state
            goal: Goal to achieve
            available_actions: Possible actions from root

        Returns:
            Best action
        """
        root = MCTSNode(root_state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: traverse tree using UCB
            while not node.is_leaf():
                node = self._select_child(node)

            # Expansion: add children if not terminal
            if not root_state.achieves_goal(goal):
                self._expand(node, available_actions, goal)

            # Simulation: evaluate the state
            value = self.scoring_module.score(node.state, goal)

            # Backpropagation: update statistics
            self._backpropagate(node, value)

        # Select best action based on visit counts
        return self._best_action(root)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.get_ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode, available_actions: List[ActionDI], goal: Goal):
        """Expand node by creating children for all actions."""
        for action in available_actions:
            new_state = action.apply(node.state)
            action_key = str(action)

            # Create child with uniform prior
            prior_prob = 1.0 / len(available_actions)
            child = MCTSNode(new_state, parent=node, action=action, prior_prob=prior_prob)
            node.children[action_key] = child

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def _best_action(self, root: MCTSNode) -> Optional[ActionDI]:
        """Select action with highest visit count."""
        best_visits = -1
        best_action = None

        for child in root.children.values():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = child.action

        return best_action


# =============================================================================
# Phase 4: Memory & Main Loop (p4 & g3)
# =============================================================================

class Experience:
    """Represents a single experience in memory."""

    def __init__(self, start_state: InternalState, action: ActionDI,
                 end_state: InternalState, score: float):
        self.start_state = start_state
        self.action = action
        self.end_state = end_state
        self.score = score

    def __repr__(self):
        return f"Experience(action={self.action}, score={self.score:.2f})"


class MemoryDB:
    """
    Storage for experiences (p4).

    This stores (state, action, result_state, score) tuples to enable learning.
    Over time, this could be used to train the scoring function.
    """

    def __init__(self, max_size: int = 1000):
        self.experiences: List[Experience] = []
        self.max_size = max_size

    def store(self, experience: Experience):
        """Store an experience in memory."""
        self.experiences.append(experience)
        # Keep only the most recent experiences
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def get_batch(self, batch_size: int) -> List[Experience]:
        """Get a random batch of experiences for training."""
        import random
        if len(self.experiences) < batch_size:
            return self.experiences.copy()
        return random.sample(self.experiences, batch_size)

    def size(self) -> int:
        """Return the number of stored experiences."""
        return len(self.experiences)

    def __repr__(self):
        return f"MemoryDB(size={self.size()}/{self.max_size})"


class LogicModel:
    """
    Main Logic Model class that ties everything together.

    This implements the g3 (project, assess, iterate) loop:
    - Project: Use search to find best action
    - Assess: Evaluate the resulting state
    - Iterate: Apply action and repeat

    Supports both Alpha-Beta and MCTS search algorithms.
    Default: Alpha-Beta pruning for deterministic, efficient search.
    """

    def __init__(self, initial_state: InternalState, goal: Goal,
                 search_type: str = 'alphabeta', search_depth: int = 3,
                 num_simulations: int = 100):
        """
        Initialize the Logic Model.

        Args:
            initial_state: Starting belief state
            goal: Goal to achieve
            search_type: 'alphabeta' (default) or 'mcts'
            search_depth: Depth for Alpha-Beta search (if used)
            num_simulations: Number of MCTS simulations (if used)
        """
        self.current_state = initial_state
        self.goal = goal
        self.search_depth = search_depth
        self.search_type = search_type
        self.num_simulations = num_simulations

        # Initialize modules
        self.scoring_module = ScoringModule()

        # Initialize search module based on type
        if search_type == 'mcts':
            self.search_module = MCTSSearch(self.scoring_module, num_simulations=num_simulations)
            self.using_mcts = True
        else:
            self.search_module = SearchModule(self.scoring_module)
            self.using_mcts = False

        self.memory = MemoryDB()

        # Statistics
        self.iteration_count = 0
        self.history: List[Tuple[InternalState, ActionDI, float]] = []

    def think_step(self, verbose: bool = False) -> bool:
        """
        Execute one step of the thinking loop (g3).

        Returns:
            True if an action was taken, False if no action available
        """
        self.iteration_count += 1

        # 1. Project: Find best action using search (p3)
        if verbose:
            print(f"\n--- Iteration {self.iteration_count} ---")
            print(f"Current state: {self.current_state}")
            print(f"Current score: {self.scoring_module.score(self.current_state, self.goal):.2f}")
            print(f"Search type: {'MCTS' if self.using_mcts else 'Alpha-Beta'}")

        # Get best action based on search type
        if self.using_mcts:
            # MCTS requires available actions
            temp_search = SearchModule(self.scoring_module)
            available_actions = temp_search.get_possible_actions(self.current_state, True)

            if not available_actions:
                if verbose:
                    print("No actions available. Stopping.")
                return False

            best_action = self.search_module.search(
                self.current_state, self.goal, available_actions
            )
        else:
            # Alpha-Beta search
            best_action = self.search_module.find_best_action(
                self.current_state, self.goal, self.search_depth
            )

        if best_action is None:
            if verbose:
                print("No action found. Stopping.")
            return False

        if verbose:
            print(f"Best action: {best_action}")

        # 2. Take action (g3)
        new_state = best_action.apply(self.current_state)
        new_score = self.scoring_module.score(new_state, self.goal)

        # 3. Store & Learn (p4)
        experience = Experience(self.current_state, best_action, new_state, new_score)
        self.memory.store(experience)

        # Update state
        self.current_state = new_state
        self.history.append((new_state, best_action, new_score))

        if verbose:
            print(f"New score: {new_score:.2f}")
            print(f"Memory size: {self.memory.size()}")

        return True

    def think(self, max_iterations: int = 10, verbose: bool = False,
              stop_on_goal: bool = True, goal_threshold: float = 0.9) -> InternalState:
        """
        Run the thinking loop for multiple iterations.

        Args:
            max_iterations: Maximum number of thinking steps
            verbose: Whether to print debug information
            stop_on_goal: Stop when goal is achieved
            goal_threshold: What score constitutes "achieving" the goal

        Returns:
            The final belief state
        """
        if verbose:
            print(f"=== Starting Logic Model ===")
            print(f"Goal: {self.goal}")
            print(f"Initial state: {self.current_state}")
            print(f"Initial score: {self.scoring_module.score(self.current_state, self.goal):.2f}")

        for i in range(max_iterations):
            # Execute one thinking step
            success = self.think_step(verbose=verbose)

            if not success:
                if verbose:
                    print("\nStopping: No more actions available")
                break

            # Check if goal is achieved
            current_score = self.scoring_module.score(self.current_state, self.goal)
            if stop_on_goal and current_score >= goal_threshold:
                if verbose:
                    print(f"\nGoal achieved! Score: {current_score:.2f}")
                break

        if verbose:
            print(f"\n=== Thinking Complete ===")
            print(f"Total iterations: {self.iteration_count}")
            print(f"Final state: {self.current_state}")
            print(f"Final score: {self.scoring_module.score(self.current_state, self.goal):.2f}")

        return self.current_state

    def get_reasoning_trace(self) -> List[str]:
        """
        Get a human-readable trace of the reasoning process.

        Returns:
            List of strings describing each step
        """
        trace = []
        for i, (state, action, score) in enumerate(self.history):
            trace.append(f"Step {i+1}: {action} -> Score: {score:.2f}")
        return trace
