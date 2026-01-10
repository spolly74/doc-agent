"""Doc Agent.

An agentic content evaluation system for technical documentation.
Evaluates articles against a 7-dimension rubric, performs JTBD coverage analysis,
validates code samples, and produces actionable reports.
"""

__version__ = "0.1.0"
__author__ = "Scott"

from doc_agent.models.enums import Dimension, RunMode, Severity

__all__ = [
    "__version__",
    "Dimension",
    "RunMode",
    "Severity",
]
