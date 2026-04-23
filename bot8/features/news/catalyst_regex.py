"""Regex-based catalyst classification.

Headlines in financial news follow extremely predictable patterns. Matching
against a curated regex set captures ~80% of the semantic value of full LLM
classification at zero cost — and the output is deterministic, inspectable,
and fast (millions of headlines per minute on CPU).

Design:
- Multi-label: a single headline can match multiple categories. The output
  is a list of tags, stored in `news_scored.catalyst_tags` as a comma-joined
  string for DuckDB simplicity.
- Case-insensitive, word-boundary-aware patterns. We err on recall over
  precision here — downstream the meta-learner can downweight noisy tags.
- Version string is stamped into `news_scored.model_version` so we can
  re-score later without losing track of which ruleset produced what.

Adding a category:
1. Add regex to `PATTERNS`.
2. Bump `RULESET_VERSION`.
3. Re-run `bot8 features news --backfill` — orchestrator will detect version
   mismatch and re-score.
"""

from __future__ import annotations

import re
from functools import lru_cache

RULESET_VERSION = "regex-v1"

# Compiled at import time. Keyed category → list of regex patterns.
# Patterns use \b word boundaries where possible to avoid sub-word matches
# (e.g. "bank" shouldn't match "banking fee" for a banking catalyst).
_RAW_PATTERNS: dict[str, list[str]] = {
    "earnings": [
        r"\b(beats?|misses?|tops?|falls short)\b.*\b(estimates?|expectations?|consensus)\b",
        r"\b(reports?|posts?|announces?)\b.*\bQ[1-4]\b",
        r"\bEPS\b",
        r"\bearnings\s+(report|release|call|beat|miss)\b",
        r"\brevenue\s+(beat|miss|rose|fell|up|down)\b",
        r"\bfiscal\s+(year|quarter)\b.*\b(results?|earnings)\b",
    ],
    "guidance": [
        r"\b(raises?|lifts?|lowers?|cuts?|reaffirms?|withdraws?|suspends?)\b.*\b(guidance|outlook|forecast)\b",
        r"\b(guidance|outlook|forecast)\b.*\b(above|below|in line|exceeds)\b",
        r"\bfull[- ]year\s+(guidance|outlook)\b",
    ],
    "m_and_a": [
        r"\b(acquires?|acquisition|acquiring)\b",
        r"\b(merger|merging|merges?)\b",
        r"\b(buyout|take[- ]?over|takeover)\b",
        r"\btender offer\b",
        r"\bto buy\b",
        r"\b(stake|equity interest)\s+(in|of)\b",
        r"\bspin[- ]?off\b",
        r"\bdivest(iture|ing|s)?\b",
    ],
    "analyst": [
        r"\b(upgrades?|upgraded|downgrades?|downgraded)\b",
        r"\bprice target\b",
        r"\binitiates?\s+(coverage|at)\b",
        r"\b(raised|lowered|cut|lifted)\s+to\s+(buy|sell|hold|outperform|underperform|overweight|underweight|neutral)\b",
        r"\b(Goldman|Morgan Stanley|JPMorgan|Citi|Barclays|Wedbush|Wells Fargo|Bank of America|BofA|Bernstein|Jefferies|Piper|Raymond James)\b.*\b(upgrade|downgrade|target|rating)\b",
    ],
    "regulatory": [
        r"\bFDA\b",
        r"\b(SEC|FTC|DOJ|FCC)\b",
        r"\bantitrust\b",
        r"\binvestigation\b",
        r"\b(probe|subpoena|lawsuit|settlement)\b",
        r"\b(approval|approves?|approved|clearance|cleared)\b",
        r"\bclinical trial\b",
        r"\b(phase [1-4]|phase [IV]+)\b",
        r"\brecall(s|ed|ing)?\b",
    ],
    "management": [
        r"\b(CEO|CFO|COO|CTO|CIO|chairman|president)\b.*\b(resigns?|steps? down|departs?|quits?|appointed|named|hired|fired|ousted)\b",
        r"\b(appoints?|names?|hires?)\b.*\b(CEO|CFO|COO|CTO|CIO|chairman|president)\b",
        r"\bexecutive\s+(change|shuffle|departure)\b",
        r"\bmanagement\s+change\b",
    ],
    "product": [
        r"\blaunch(es|ed|ing)?\b",
        r"\bunveils?\b",
        r"\bannounce(s|d|ment)\b.*\b(new|first)\b",
        r"\breveals?\b.*\b(product|device|model|service)\b",
        r"\brollout\b",
    ],
    "legal": [
        r"\blawsuit\b",
        r"\b(sues?|sued|suing)\b",
        r"\bcourt\s+(ruling|decision|order|filing)\b",
        r"\bpatent\s+(infringement|dispute|lawsuit)\b",
        r"\bclass[- ]action\b",
    ],
    "dividend_buyback": [
        r"\b(dividend|div)\s+(increase|raise|hike|cut|suspension|special)\b",
        r"\b(raises?|lifts?|cuts?|suspends?)\s+(the\s+)?dividend\b",
        r"\b(share|stock)\s+buyback\b",
        r"\brepurchase\s+program\b",
        r"\bauthoriz(es?|ed)?\s+.*\bbuyback\b",
    ],
    "insider": [
        r"\binsider\s+(buying|selling|transaction|purchase|sale)\b",
        r"\b(CEO|CFO|director|executive)\s+(buys?|sells?|bought|sold|purchas)\b",
        r"\bForm 4\b",
    ],
    "macro": [
        r"\b(Fed|Federal Reserve|FOMC)\b",
        r"\b(interest rate|rate hike|rate cut|rate decision)\b",
        r"\b(CPI|PPI|inflation|unemployment|payroll|jobs report)\b",
        r"\btariff(s)?\b",
        r"\btrade war\b",
        r"\brecession\b",
    ],
    "bankruptcy": [
        r"\bChapter\s+(7|11)\b",
        r"\bbankruptcy\b",
        r"\bdefault(s|ed|ing)?\b",
        r"\bgoing concern\b",
    ],
}


@lru_cache(maxsize=1)
def _compiled_patterns() -> dict[str, list[re.Pattern[str]]]:
    """Compile once. re.IGNORECASE for everything."""
    return {
        cat: [re.compile(p, re.IGNORECASE) for p in patterns]
        for cat, patterns in _RAW_PATTERNS.items()
    }


def classify(text: str) -> list[str]:
    """Return the list of catalyst tags matched by `text`.

    Empty list if no pattern matches. Order is stable (dict insertion order).

    >>> classify("Apple beats Q3 earnings estimates, raises guidance")
    ['earnings', 'guidance']
    >>> classify("Goldman upgrades NVDA, raises price target to $1200")
    ['analyst']
    >>> classify("SEC opens investigation into Tesla autopilot claims")
    ['regulatory']
    >>> classify("some random headline")
    []
    """
    if not text:
        return []
    matched: list[str] = []
    for cat, patterns in _compiled_patterns().items():
        if any(p.search(text) for p in patterns):
            matched.append(cat)
    return matched


def classify_to_tag_string(text: str) -> str:
    """Return comma-joined tags (for DuckDB VARCHAR column). Empty string if none."""
    return ",".join(classify(text))
