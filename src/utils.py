"""Small shared utility functions for the Umba fraud solution."""

from __future__ import annotations


def risk_band(score: float) -> str:
    """Convert a fraud probability into an operational risk band."""
    if score >= 0.90:
        return "Critical"
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"
