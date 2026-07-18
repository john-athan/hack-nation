"""Output schema — the contract the demo, API, and tests all agree on.

Frozen dataclasses instead of pydantic: lean (no extra runtime dep) and the shape
is small and stable. `to_dict` gives the exact per-prediction JSON the PDF asks for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class DrugPrediction:
    """One antibiotic verdict for one genome."""

    antibiotic: str
    call: str  # constants.CALL_*
    confidence: float  # 0..1; for the T1 dummy this is a coarse rule-based value
    evidence_category: str  # constants.EVIDENCE_*
    supporting_genes: list[str]  # determinants that drove the call; [] if none
    target_present: bool  # is the drug's molecular target gene present in the genome?

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenomeReport:
    """The full per-genome response: one prediction per drug + a mandatory disclaimer."""

    genome_id: str
    predictions: list[DrugPrediction] = field(default_factory=list)
    disclaimer: str = (
        "Decision support only — not a diagnosis. "
        "Confirm every result with standard laboratory susceptibility testing."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "genome_id": self.genome_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "disclaimer": self.disclaimer,
        }
