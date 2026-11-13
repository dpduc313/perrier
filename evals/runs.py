"""Saved eval runs, and the ADR-0006 switch-over gate between two of them.

Every `python -m evals` run saves its results under `evals/runs/` (kept out of git);
`python -m evals --compare BASELINE CANDIDATE` checks a candidate setup against a baseline.
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from evals.harness import Result

RUNS_DIR = Path(__file__).parent / "runs"

# ADR-0006's gate counts these hard checks; the candidate may pass at most
# MAX_FEWER_CHECKS fewer of them than the baseline.
GATE_CHECKS = ("routing", "Citations")
MAX_FEWER_CHECKS = 1
# ADR-0006: and its average judge score may be at most this much lower, relative to the
# baseline's ("within about 10%").
MAX_JUDGE_DROP = 0.1
# A Turn's judge score moving by this much (one point on the judge's 1-5 scale) counts as
# a different result.
JUDGE_POINT = 0.25


@dataclass(frozen=True)
class TurnRecord:
    """One eval Turn's outcome in a saved run."""

    id: str
    routing: bool
    citations: bool
    confirmations: bool
    quality: float | None  # 0..1 from the judge; None if it gave no usable score
    cost_usd: float
    error: str | None = None

    @property
    def check_results(self) -> dict[str, bool]:
        """Every hard check, by name: GATE_CHECKS, and Confirmations (shown, not gated)."""
        return {
            "routing": self.routing,
            "Citations": self.citations,
            "Confirmations": self.confirmations,
        }

    @property
    def gate_checks(self) -> int:
        """The GATE_CHECKS passed."""
        return sum(self.check_results[name] for name in GATE_CHECKS)


@dataclass(frozen=True)
class Run:
    when: str  # ISO 8601, with the offset
    commit: str
    settings: dict[str, str | None]  # who answers: `models.who_answers()`
    turns: list[TurnRecord]


@dataclass(frozen=True)
class Comparison:
    turns: int  # in both runs: the ones compared
    baseline_checks: int  # GATE_CHECKS passed
    candidate_checks: int
    baseline_quality: float | None  # average judge score, None without any
    candidate_quality: float | None
    differing: list[tuple[TurnRecord, TurnRecord]]  # (baseline, candidate), in run order
    left_out: int  # Turns in only one of the runs
    baseline_cost: float  # USD for the whole run
    candidate_cost: float

    @property
    def judge_change(self) -> float | None:
        """The candidate's average judge score relative to the baseline's (-0.1 is 10%
        lower), rounded so a score exactly at the threshold counts as at it."""
        if not self.baseline_quality or self.candidate_quality is None:
            return None
        return round(self.candidate_quality / self.baseline_quality - 1, 4)

    @property
    def failures(self) -> list[str]:
        """Why the candidate fails the ADR-0006 gate; empty if it passes. Without judge
        scores to compare, it can't pass."""
        reasons = []
        if self.candidate_checks < self.baseline_checks - MAX_FEWER_CHECKS:
            reasons.append(f"more than {MAX_FEWER_CHECKS} fewer hard checks passed")
        if self.judge_change is None:
            reasons.append("no judge scores to compare")
        elif self.judge_change < -MAX_JUDGE_DROP:
            reasons.append(f"judge score more than {MAX_JUDGE_DROP:.0%} lower")
        return reasons

    @property
    def passed(self) -> bool:
        return not self.failures


def save(
    results: list[Result], settings: dict[str, str | None], directory: Path = RUNS_DIR
) -> Path:
    """Save a run's results with when it ran, the commit and `settings`; returns the file."""
    now = datetime.now().astimezone()
    commit = _commit()
    run = Run(
        when=now.isoformat(timespec="seconds"),
        commit=commit,
        settings=settings,
        turns=[_record(r) for r in results],
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%Y-%m-%dT%H%M%S}-{commit}.json"
    path.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
    return path


def load(path: Path) -> Run:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Run(**{**data, "turns": [TurnRecord(**t) for t in data["turns"]]})


def _record(result: Result) -> TurnRecord:
    return TurnRecord(
        id=result.turn.id,
        routing=result.routing_ok,
        citations=result.citations_ok,
        confirmations=result.confirmations_ok,
        quality=result.quality,
        cost_usd=result.cost_usd,
        error=result.error,
    )


def _commit() -> str:
    """The checkout's commit, "+changes" if it has uncommitted ones."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=RUNS_DIR.parent
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        return f"{commit}+changes" if git("status", "--porcelain") else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def compare(baseline: Run, candidate: Run) -> Comparison:
    """`candidate` against `baseline` on the ADR-0006 gate, over the Turns both ran."""
    candidate_by_id = {t.id: t for t in candidate.turns}
    pairs = [(t, candidate_by_id[t.id]) for t in baseline.turns if t.id in candidate_by_id]
    baseline_shared = [b for b, _ in pairs]
    candidate_shared = [c for _, c in pairs]
    return Comparison(
        turns=len(pairs),
        baseline_checks=sum(t.gate_checks for t in baseline_shared),
        candidate_checks=sum(t.gate_checks for t in candidate_shared),
        baseline_quality=_average_quality(baseline_shared),
        candidate_quality=_average_quality(candidate_shared),
        differing=[(b, c) for b, c in pairs if _difference(b, c)],
        left_out=len(baseline.turns) + len(candidate.turns) - 2 * len(pairs),
        baseline_cost=sum(t.cost_usd for t in baseline.turns),
        candidate_cost=sum(t.cost_usd for t in candidate.turns),
    )


def _difference(baseline: TurnRecord, candidate: TurnRecord) -> str:
    """How the candidate's result differs from the baseline's; empty if it doesn't."""
    was = baseline.check_results
    changes = [
        f"{name} {_check(was[name])} -> {_check(now)}"
        for name, now in candidate.check_results.items()
        if now != was[name]
    ]
    if _judge_differs(baseline.quality, candidate.quality):
        changes.append(f"judge {_score(baseline.quality)} -> {_score(candidate.quality)}")
    if changes and candidate.error:
        changes.append(f"error: {candidate.error}")
    return "; ".join(changes)


def _judge_differs(was: float | None, now: float | None) -> bool:
    if was is None or now is None:
        return was != now
    return abs(was - now) >= JUDGE_POINT


def _average_quality(turns: list[TurnRecord]) -> float | None:
    scores = [t.quality for t in turns if t.quality is not None]
    return sum(scores) / len(scores) if scores else None


def report(baseline: Run, candidate: Run) -> str:
    """What `--compare` prints: both runs side by side and the gate's verdict."""
    c = compare(baseline, candidate)
    total = c.turns * len(GATE_CHECKS)
    change = "" if c.judge_change is None else f"{c.judge_change:+.1%}"
    lines = [
        f"Baseline:  {_describe(baseline)}",
        f"Candidate: {_describe(candidate)}",
        f"Compared on {c.turns} Turns; {c.left_out} in only one run left out.",
        "",
        _row("", "Baseline", "Candidate"),
        _row(
            "Routing and Citation checks",
            f"{c.baseline_checks}/{total}",
            f"{c.candidate_checks}/{total}",
        ),
        _row(
            "Average judge score", _score(c.baseline_quality), _score(c.candidate_quality), change
        ),
        _row("Cost (whole run)", f"${c.baseline_cost:.4f}", f"${c.candidate_cost:.4f}"),
        "",
    ]
    if c.differing:
        lines.append("Turns whose result differs:")
        lines += [f"  {b.id}: {_difference(b, d)}" for b, d in c.differing]
    else:
        lines.append("No Turn's result differs.")
    verdict = f"FAIL ({'; '.join(c.failures)})" if c.failures else "PASS"
    lines += ["", f"Verdict: {verdict}"]
    return "\n".join(lines)


def _row(label: str, baseline: str, candidate: str, change: str = "") -> str:
    return f"{label:29}{baseline:11}{candidate:11}{change}".rstrip()


def _describe(run: Run) -> str:
    settings = ", ".join(f"{name} {value or 'none'}" for name, value in run.settings.items())
    return f"{run.when}  {run.commit}  {settings}".rstrip()


def _score(quality: float | None) -> str:
    return "n/a" if quality is None else f"{quality:.2f}"


def _check(passed: bool) -> str:
    return "pass" if passed else "FAIL"
