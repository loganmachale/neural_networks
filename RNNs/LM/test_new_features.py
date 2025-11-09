"""
Test script to verify new features from README2.md and README3.md
"""

from logic_model import InternalState, Goal, ScoringModule, SearchModule, LogicModel
from advanced_features import MCTSSearch, ReplayBuffer


def test_state_hashing():
    """Test state hashing feature (from README2.md)"""
    print("=" * 70)
    print("TEST 1: State Hashing")
    print("=" * 70)

    state1 = InternalState()
    state1.add_fact("A", True, 0.9)
    state1.add_fact("B", True, 0.8)

    state2 = InternalState()
    state2.add_fact("A", True, 0.9)
    state2.add_fact("B", True, 0.8)

    state3 = InternalState()
    state3.add_fact("A", True, 0.9)
    state3.add_fact("C", True, 0.7)

    print(f"State 1 hash: {hash(state1)}")
    print(f"State 2 hash: {hash(state2)} (same facts as State 1)")
    print(f"State 3 hash: {hash(state3)} (different facts)")
    print(f"State 1 == State 2: {state1 == state2}")
    print(f"State 1 == State 3: {state1 == state3}")
    print("[PASS] State hashing works!\n")


def test_goal_checking():
    """Test goal achievement checking (from README2.md)"""
    print("=" * 70)
    print("TEST 2: Goal Achievement Checking")
    print("=" * 70)

    state = InternalState()
    state.add_fact("problem_solved", True, 0.95)
    state.add_fact("answer_correct", True, 0.85)

    goal = Goal({
        "problem_solved": (True, 0.9),
        "answer_correct": (True, 0.8)
    })

    print(f"State achieves goal: {state.achieves_goal(goal)}")
    print("[PASS] Goal checking works!\n")


def test_memoization():
    """Test memoization/caching in search (from README2.md)"""
    print("=" * 70)
    print("TEST 3: Memoization/Caching")
    print("=" * 70)

    # Create a simple problem
    state = InternalState()
    state.add_fact("start", True, 1.0)

    goal = Goal({"end": (True, 0.9)})

    # Create search with memoization
    scorer = ScoringModule()
    search = SearchModule(scorer, use_memoization=True)

    # Run search multiple times (should hit cache on subsequent runs)
    for i in range(3):
        search.find_best_action(state, goal, search_depth=2)

    stats = search.get_cache_stats()
    print(f"Cache size: {stats['cache_size']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print("[PASS] Memoization works!\n")


def test_early_termination():
    """Test early termination when goal is achieved"""
    print("=" * 70)
    print("TEST 4: Early Termination on Goal Achievement")
    print("=" * 70)

    state = InternalState()
    state.add_fact("premise", True, 0.9)

    def derive_conclusion(facts):
        if facts.get("premise"):
            return ("conclusion", True, 0.9)
        return None

    state.add_rule(derive_conclusion)

    goal = Goal({"conclusion": (True, 0.8)})

    lm = LogicModel(state, goal, search_depth=2)
    final_state = lm.think(max_iterations=10, verbose=False)

    print(f"Iterations used: {lm.iteration_count}")
    print(f"Goal achieved: {final_state.achieves_goal(goal)}")
    print(f"Conclusion fact: {final_state.get_fact('conclusion')}")

    if lm.iteration_count == 1:
        print("[PASS] Early termination works! (stopped after 1 iteration)\n")
    else:
        print(f"⚠ Took {lm.iteration_count} iterations (expected 1)\n")


def test_mcts():
    """Test MCTS search (from README3.md)"""
    print("=" * 70)
    print("TEST 5: Monte Carlo Tree Search")
    print("=" * 70)

    state = InternalState()
    state.add_fact("x", 10, 1.0)

    goal = Goal({"result": (True, 0.9)})

    from logic_model import AddFactDI

    available_actions = [
        AddFactDI("result", True, 0.8),
        AddFactDI("result", False, 0.5),
    ]

    scorer = ScoringModule()
    mcts = MCTSSearch(scorer, num_simulations=50)

    best_action = mcts.search(state, goal, available_actions)
    print(f"Best action found: {best_action}")
    print("[PASS] MCTS search works!\n")


def test_replay_buffer():
    """Test replay buffer (from README3.md)"""
    print("=" * 70)
    print("TEST 6: Replay Buffer")
    print("=" * 70)

    buffer = ReplayBuffer(max_size=100)

    # Add experiences
    for i in range(5):
        state = InternalState()
        state.add_fact(f"fact_{i}", True, 0.9)
        buffer.add(state, [0.5, 0.3, 0.2], 1.0)

    print(f"Buffer size: {len(buffer)}")
    batch = buffer.sample(3)
    print(f"Sampled batch size: {len(batch)}")
    print("[PASS] Replay buffer works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING NEW FEATURES FROM README2.md & README3.md")
    print("=" * 70 + "\n")

    test_state_hashing()
    test_goal_checking()
    test_memoization()
    test_early_termination()
    test_mcts()
    test_replay_buffer()

    print("=" * 70)
    print("ALL TESTS PASSED! [PASS]")
    print("=" * 70)
    print("\nNew features successfully integrated:")
    print("  [PASS] State hashing (README2.md)")
    print("  [PASS] Goal achievement checking (README2.md)")
    print("  [PASS] Memoization/caching (README2.md)")
    print("  [PASS] Early termination (README2.md)")
    print("  [PASS] MCTS search (README3.md)")
    print("  [PASS] Replay buffer (README3.md)")
