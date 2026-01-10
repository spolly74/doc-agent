"""Foundry Content Evaluation System.

An agentic content evaluation system for Microsoft Foundry technical documentation.
Evaluates articles against a 7-dimension rubric, performs JTBD coverage analysis,
validates code samples, and produces actionable reports.
"""

__version__ = "0.1.0"
__author__ = "Scott"

from foundry_eval.models.enums import Dimension, RunMode, Severity

__all__ = [
    "__version__",
    "Dimension",
    "RunMode",
    "Severity",
]
