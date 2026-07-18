"""AuditReport: the result object returned by resample_audit.audit()."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class AuditReport:
    """Outcome of a validity + information-gain audit of one resampler.

    The standard (see paper): a generator earns its place only if
    (1) its synthetic points are valid minority — ER_split (a consistent
        estimate of 1 - V(G)) is low, and
    (2) it adds information a trivial decision-rule change cannot —
        I(G) = F1 over the best of {no_resample, class_weight,
        threshold_move} is positive.
    """
    resampler: str
    n_synth: int
    # validity
    er_naive: float
    er_naive_ci: tuple
    er_split: float
    er_split_ci: tuple
    er_split_std: float
    splits_used: int
    bias_gain: float            # er_split - er_naive (what the naive test hid)
    # information gain
    info_gain: float | None = None
    best_baseline: str | None = None
    delta_pr_auc: float | None = None
    delta_roc_auc: float | None = None
    delta_brier: float | None = None
    delta_ece: float | None = None
    arms: dict = field(default_factory=dict)
    # verdict
    er_max: float = 0.10
    metric: str = "hassanat"

    @property
    def valid(self) -> bool:
        """Validity clause: upper Wilson bound of ER_split below er_max."""
        return bool(self.er_split_ci[1] < self.er_max)

    @property
    def adds_information(self) -> bool | None:
        return None if self.info_gain is None else bool(self.info_gain > 0)

    @property
    def passes_standard(self) -> bool | None:
        if self.info_gain is None:
            return None
        return bool(self.valid and self.info_gain > 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(valid=self.valid, adds_information=self.adds_information,
                 passes_standard=self.passes_standard)
        return d

    def __str__(self) -> str:
        ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
        L = [f"resample-audit report — {self.resampler}",
             "=" * 46,
             f"synthetic minority points        {self.n_synth}",
             f"ER_naive  (parent-retained ref)  {self.er_naive:.3f} "
             f"{ci(self.er_naive_ci)}",
             f"ER_split  (de-biased, est 1-V)   {self.er_split:.3f} "
             f"{ci(self.er_split_ci)}  (std {self.er_split_std:.3f}, "
             f"{self.splits_used} splits)",
             f"hidden bias (split - naive)      {self.bias_gain:+.3f}",
             f"validity (ER_split hi < {self.er_max:.2f})    "
             f"{'PASS' if self.valid else 'FAIL'}"]
        if self.info_gain is not None:
            ov = self.arms.get("oversample", {})
            nb = self.arms.get(self.best_baseline, {})
            L += ["-" * 46,
                  f"best trivial baseline            {self.best_baseline} "
                  f"(F1 {nb.get('f1', float('nan')):.3f})",
                  f"oversampled F1                   "
                  f"{ov.get('f1', float('nan')):.3f}",
                  f"information gain I(G)            {self.info_gain:+.3f}"
                  f"   -> {'PASS' if self.info_gain > 0 else 'FAIL'}",
                  f"delta PR-AUC / ROC-AUC           {self.delta_pr_auc:+.4f}"
                  f" / {self.delta_roc_auc:+.4f}",
                  f"delta Brier / ECE (>0 = worse)   {self.delta_brier:+.4f}"
                  f" / {self.delta_ece:+.4f}",
                  "-" * 46,
                  f"STANDARD (valid AND informative) "
                  f"{'PASS' if self.passes_standard else 'FAIL'}"]
        return "\n".join(L)
