"""FinBERT sentiment scoring.

Uses `ProsusAI/finbert` from HuggingFace: a BERT model fine-tuned on the
Financial PhraseBank for 3-class sentiment (positive / neutral / negative).

Design:
- Lazy model load: the first call to `FinBertScorer.score()` downloads and
  caches the model (~440MB). Subsequent runs load from the HF cache instantly.
- Device auto-select: MPS (Apple Silicon) > CUDA > CPU. On M-series Macs MPS
  gives ~10-20× speedup over CPU for BERT inference.
- Batched inference: pass a list of strings, get batched tensor inference.
  Default batch=32, tune up on M4 Pro/Max.
- `signed_score` maps 3-class logits to a single scalar in [-1, +1]:
    +P(positive) − P(negative)
  This is what feeds the meta-learner downstream (monotonic + differentiable).
- `confidence` = max class probability (how sure the model is of its top pick).
- Deterministic — `torch.no_grad()` and eval mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

MODEL_VERSION = "finbert-prosus-v1"

# Separator between FinBERT scorer version and catalyst ruleset version in
# `news_scored.model_version`. Stored as a single string so we know exactly
# which pair produced each row.
MODEL_VERSION_FORMAT = "{finbert}+{catalyst}"


@dataclass(frozen=True, slots=True)
class FinBertOutput:
    label: str            # 'positive' | 'neutral' | 'negative'
    signed_score: float   # P(pos) − P(neg), in [-1, +1]
    confidence: float     # max probability, in [0, 1]


class FinBertScorer:
    """Batched FinBERT wrapper. Thread-hostile — create one per process."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        # Imports are inside __init__ so users who only need the regex module
        # don't pay the transformers import cost (which drags torch with it).
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from bot8.config import get_settings
        settings = get_settings()

        self._torch = torch
        self.model_name = model_name or settings.finbert_model
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = device or self._select_device()
        logger.info("FinBERT loading {} onto {}", self.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT's label order (id → label). Set on the HF config.
        # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral (verified).
        self.id2label: dict[int, str] = self.model.config.id2label

        # Pre-compute label positions for the signed_score formula.
        label_to_id = {v.lower(): k for k, v in self.id2label.items()}
        self._pos_id = label_to_id["positive"]
        self._neg_id = label_to_id["negative"]

    def _select_device(self) -> str:
        t = self._torch
        if t.cuda.is_available():
            return "cuda"
        if hasattr(t.backends, "mps") and t.backends.mps.is_available():
            return "mps"
        return "cpu"

    def score(self, texts: list[str]) -> list[FinBertOutput]:
        """Score a list of headlines. Returns one FinBertOutput per input."""
        if not texts:
            return []

        torch = self._torch
        results: list[FinBertOutput] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            # Move to CPU once, convert to Python floats
            probs_cpu = probs.detach().cpu().numpy()
            for row in probs_cpu:
                top_id = int(row.argmax())
                results.append(
                    FinBertOutput(
                        label=self.id2label[top_id].lower(),
                        signed_score=float(row[self._pos_id] - row[self._neg_id]),
                        confidence=float(row[top_id]),
                    )
                )

        return results

    def score_one(self, text: str) -> FinBertOutput:
        """Convenience: single-headline scoring."""
        return self.score([text])[0]


# ---------------------------------------------------------------------------
# Module-level convenience — lazy singleton for CLI / notebook use.
# ---------------------------------------------------------------------------
_singleton: FinBertScorer | None = None


def get_scorer(**kwargs: Any) -> FinBertScorer:
    """Cached per-process FinBertScorer. Args only honoured on first call."""
    global _singleton
    if _singleton is None:
        _singleton = FinBertScorer(**kwargs)
    return _singleton
