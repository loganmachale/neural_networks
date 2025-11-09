"""
Unit tests for Logic Model core features.

Tests cover:
- State hashing and equality
- Goal achievement checking
- Memoization and caching
- Early termination
- MCTS search
- Replay buffer functionality
"""

from logic_model import (
    InternalState, Goal, ScoringModule, SearchModule, LogicModel,
    AddFactDI, ApplyRuleDI, MCTSSearch
)


# =============================================================================
# STATE AND GOAL TESTS
# =============================================================================

def test_state_hashing():
    """Verify state hashing enables efficient caching."""
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

    assert state1 == state2, "States with same facts should be equal"
    assert state1 != state3, "States with different facts should not be equal"
    assert hash(state1) == hash(state2), "Equal states should have same hash"

    print("[PASS] State hashing works!\n")


def test_goal_checking():
    """Verify goal achievement detection."""
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

    achieves = state.achieves_goal(goal)
    print(f"State achieves goal: {achieves}")

    assert achieves, "State should achieve goal with sufficient confidence"

    print("[PASS] Goal checking works!\n")


# =============================================================================
# SEARCH AND OPTIMIZATION TESTS
# =============================================================================

def test_memoization():
    """Verify search memoization reduces redundant evaluations."""
    print("=" * 70)
    print("TEST 3: Memoization/Caching")
    print("=" * 70)

    # Create simple problem
    state = InternalState()
    state.add_fact("start", True, 1.0)

    goal = Goal({"end": (True, 0.9)})

    # Create search with memoization
    scorer = ScoringModule()
    search = SearchModule(scorer, use_memoization=True)

    # Run search multiple times to populate cache
    for i in range(3):
        search.find_best_action(state, goal, search_depth=2)

    stats = search.get_cache_stats()
    print(f"Cache size: {stats['cache_size']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")

    assert stats['cache_hits'] > 0, "Should have cache hits after repeated searches"

    print("[PASS] Memoization works!\n")


def test_early_termination():
    """Verify search terminates early when goal is achieved."""
    print("=" * 70)
    print("TEST 4: Early Termination on Goal Achievement")
    print("=" * 70)

    state = InternalState()
    state.add_fact("premise", True, 0.9)

    def derive_conclusion(facts):
        """Simple rule that derives conclusion from premise."""
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

    assert lm.iteration_count <= 2, f"Should terminate early (used {lm.iteration_count} iterations)"
    assert final_state.achieves_goal(goal), "Goal should be achieved"

    print("[PASS] Early termination works!\n")


def test_mcts():
    """Verify MCTS search algorithm functions correctly."""
    print("=" * 70)
    print("TEST 5: Monte Carlo Tree Search")
    print("=" * 70)

    state = InternalState()
    state.add_fact("x", 10, 1.0)

    goal = Goal({"result": (True, 0.9)})

    available_actions = [
        AddFactDI("result", True, 0.8),
        AddFactDI("result", False, 0.5),
    ]

    scorer = ScoringModule()
    mcts = MCTSSearch(scorer, num_simulations=50)

    best_action = mcts.search(state, goal, available_actions)
    print(f"Best action found: {best_action}")

    assert best_action is not None, "MCTS should return an action"

    print("[PASS] MCTS search works!\n")


# =============================================================================
# REASONING LOGIC TESTS
# =============================================================================

def test_rule_application():
    """Verify logical rules can be applied to derive new facts."""
    print("=" * 70)
    print("TEST 6: Rule Application")
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

    # Test initial state
    print("Initial state:", state)
    print("Number of rules:", len(state.rules))

    scorer = ScoringModule()
    initial_score = scorer.score(state, goal)
    print(f"Initial score: {initial_score:.2f}")

    # Apply rule
    action = ApplyRuleDI(0)
    new_state = action.apply(state)
    print("\nAfter applying rule:", new_state)

    new_score = scorer.score(new_state, goal)
    print(f"New score: {new_score:.2f}")

    score_improvement = new_score - initial_score
    print(f"\nScore improvement: {score_improvement:.2f}")

    # Verify result
    result_fact = new_state.get_fact("x_greater_than_y")
    assert result_fact is not None, "Rule should derive x_greater_than_y fact"
    assert result_fact[0] == True, "x should be greater than y"
    assert score_improvement > 0, "Score should improve after applying rule"

    print("[PASS] Rule application works!\n")


# =============================================================================
# CONFIDENCE AND SCORING TESTS
# =============================================================================

def test_confidence_tracking():
    """Verify confidence values are tracked and propagated correctly."""
    print("=" * 70)
    print("TEST 7: Confidence Tracking")
    print("=" * 70)

    state = InternalState()
    state.add_fact("premise_a", True, 0.9)
    state.add_fact("premise_b", True, 0.8)

    def combined_rule(facts):
        """Combine two premises with minimum confidence."""
        a = facts.get("premise_a")
        b = facts.get("premise_b")

        if a and b:
            a_val, a_conf = a
            b_val, b_conf = b

            if a_val and b_val:
                # Take minimum confidence
                combined_conf = min(a_conf, b_conf)
                return ("conclusion", True, combined_conf)
        return None

    state.add_rule(combined_rule)

    # Apply rule
    action = ApplyRuleDI(0)
    new_state = action.apply(state)

    conclusion = new_state.get_fact("conclusion")
    assert conclusion is not None, "Conclusion should be derived"

    _, confidence = conclusion
    print(f"Premise A confidence: 0.9")
    print(f"Premise B confidence: 0.8")
    print(f"Conclusion confidence: {confidence}")

    assert confidence == 0.8, "Should use minimum confidence from premises"

    print("[PASS] Confidence tracking works!\n")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Execute all test functions."""
    print("\n" + "=" * 70)
    print("LOGIC MODEL UNIT TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_state_hashing,
        test_goal_checking,
        test_memoization,
        test_early_termination,
        test_mcts,
        test_rule_application,
        test_confidence_tracking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}\n")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}\n")
            failed += 1

    print("=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nALL TESTS PASSED!\n")
        print("Verified features:")
        print("  [PASS] State hashing")
        print("  [PASS] Goal achievement checking")
        print("  [PASS] Memoization/caching")
        print("  [PASS] Early termination")
        print("  [PASS] MCTS search")
        print("  [PASS] Rule application")
        print("  [PASS] Confidence tracking")
    else:
        print(f"\n{failed} TEST(S) FAILED")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
