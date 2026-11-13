"""Saved eval runs and the ADR-0006 switch-over gate between two of them."""

from evals import runs
from evals.harness import EvalTurn, Result


def turn(id, routing=True, citations=True, confirmations=True, quality=1.0, cost=0.0):
    return runs.TurnRecord(
        id=id,
        routing=routing,
        citations=citations,
        confirmations=confirmations,
        quality=quality,
        cost_usd=cost,
    )


def run(*turns):
    return runs.Run(
        when="2026-10-19T10:00:00+07:00",
        commit="abc1234",
        settings={},
        turns=list(turns),
    )


def test_one_fewer_hard_check_passed_still_passes_the_gate():
    baseline = run(turn("a"), turn("b"), turn("c"))
    candidate = run(turn("a"), turn("b", citations=False), turn("c"))

    comparison = runs.compare(baseline, candidate)

    assert (comparison.baseline_checks, comparison.candidate_checks) == (6, 5)
    assert comparison.passed


def test_two_fewer_hard_checks_passed_fails_the_gate():
    baseline = run(turn("a"), turn("b"), turn("c"))
    candidate = run(turn("a", routing=False), turn("b", citations=False), turn("c"))

    assert not runs.compare(baseline, candidate).passed


def test_a_judge_score_10_percent_lower_still_passes_the_gate():
    baseline = run(turn("a", quality=1.0), turn("b", quality=0.6))  # average 0.8
    candidate = run(turn("a", quality=0.94), turn("b", quality=0.5))  # average 0.72

    comparison = runs.compare(baseline, candidate)

    assert comparison.judge_change == -0.1
    assert comparison.passed


def test_a_judge_score_more_than_10_percent_lower_fails_the_gate():
    baseline = run(turn("a", quality=1.0), turn("b", quality=0.6))  # average 0.8
    candidate = run(turn("a", quality=0.92), turn("b", quality=0.5))  # average 0.71

    assert not runs.compare(baseline, candidate).passed


def test_without_judge_scores_the_gate_cant_pass():
    baseline = run(turn("a", quality=None))
    candidate = run(turn("a", quality=None))

    comparison = runs.compare(baseline, candidate)

    assert comparison.judge_change is None
    assert not comparison.passed


def test_runs_over_different_turns_are_compared_on_the_turns_they_share():
    baseline = run(turn("a"), turn("b"), turn("only-in-baseline", quality=0.0))
    candidate = run(turn("a"), turn("b"), turn("only-in-candidate", routing=False))

    comparison = runs.compare(baseline, candidate)

    assert (comparison.baseline_checks, comparison.candidate_checks) == (4, 4)
    assert comparison.judge_change == 0.0
    assert comparison.left_out == 2


def test_the_turns_whose_result_differs_are_named_with_each_runs_cost():
    baseline = run(turn("a", cost=0.01), turn("b", cost=0.02), turn("c", quality=1.0))
    candidate = run(turn("a"), turn("b", citations=False), turn("c", quality=0.5))

    comparison = runs.compare(baseline, candidate)

    assert [t.id for t, _ in comparison.differing] == ["b", "c"]
    assert comparison.baseline_cost == 0.03
    assert comparison.candidate_cost == 0.0


def test_a_lost_confirmation_is_named_but_isnt_one_of_the_gates_checks():
    baseline = run(turn("send-email"))
    candidate = run(turn("send-email", confirmations=False))

    comparison = runs.compare(baseline, candidate)

    assert (comparison.baseline_checks, comparison.candidate_checks) == (2, 2)
    assert comparison.passed
    assert "send-email: Confirmations pass -> FAIL" in runs.report(baseline, candidate)


def test_a_small_change_in_a_turns_judge_score_isnt_a_different_result():
    baseline = run(turn("a", quality=1.0))
    candidate = run(turn("a", quality=0.8))

    assert runs.compare(baseline, candidate).differing == []


def test_a_saved_run_loads_back_with_its_settings_and_each_turns_checks(tmp_path):
    budget = EvalTurn("q3-budget", "How big is my Q3 budget?", ("knowledge",), ("budget.md",))
    hello = EvalTurn("hello", "Hello there", ())
    results = [
        Result(budget, consulted=["knowledge"], quality=0.75, cost_usd=0.002),
        Result(hello, consulted=["knowledge"], quality=None),
        Result(hello, error="TimeoutError: slow"),
    ]
    settings = {"chat_model": "gpt-5.4-mini", "inet_qwen_for": None}

    path = runs.save(results, settings, tmp_path)
    saved = runs.load(path)

    assert path.parent == tmp_path
    assert saved.settings == settings
    assert saved.commit  # the checkout's commit
    assert saved.turns == [
        turn("q3-budget", quality=0.75, cost=0.002),
        turn("hello", routing=False, quality=None),
        runs.TurnRecord("hello", False, False, False, None, 0.0, error="TimeoutError: slow"),
    ]


def test_the_report_gives_the_counts_judge_difference_differing_turns_costs_and_verdict():
    baseline = run(turn("a", cost=0.0125), turn("b", quality=0.75), turn("gone"))
    candidate = run(turn("a"), turn("b", citations=False, quality=0.5))

    report = runs.report(baseline, candidate)

    assert "Compared on 2 Turns; 1 in only one run left out." in report
    assert "Routing and Citation checks  4/4        3/4" in report
    assert "Average judge score          0.88       0.75       -14.3%" in report
    assert "Cost (whole run)             $0.0125    $0.0000" in report
    assert "  b: Citations pass -> FAIL; judge 0.75 -> 0.50" in report
    assert report.endswith("Verdict: FAIL (judge score more than 10% lower)")
