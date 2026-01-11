"""Enumerations for the Doc Agent evaluation system."""

from enum import Enum


class RunMode(str, Enum):
    """Evaluation run modes."""

    FULL = "full"
    TIERED = "tiered"
    JTBD = "jtbd"
    SAMPLES = "samples"


class Dimension(str, Enum):
    """Evaluation dimensions from the rubric."""

    PATTERN_COMPLIANCE = "pattern_compliance"
    DEV_FOCUS = "dev_focus"
    CODE_QUALITY = "code_quality"
    CODE_COMPLETENESS = "code_completeness"
    STRUCTURE = "structure"
    ACCURACY = "accuracy"
    BRANDING_COMPLIANCE = "branding_compliance"
    JTBD_ALIGNMENT = "jtbd_alignment"


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ContentPattern(str, Enum):
    """Documentation content patterns (e.g., from ms.topic metadata)."""

    HOW_TO = "how-to"
    QUICKSTART = "quickstart"
    TUTORIAL = "tutorial"
    CONCEPT = "conceptual"
    REFERENCE = "reference"
    OVERVIEW = "overview"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    SAMPLE = "sample"
    UNKNOWN = "unknown"

    @classmethod
    def from_ms_topic(cls, ms_topic: str | None) -> "ContentPattern":
        """Convert ms.topic value to ContentPattern.

        Args:
            ms_topic: The ms.topic frontmatter value.

        Returns:
            Corresponding ContentPattern enum value.
        """
        if ms_topic is None:
            return cls.UNKNOWN

        ms_topic = ms_topic.lower().strip()

        mapping = {
            "how-to": cls.HOW_TO,
            "howto": cls.HOW_TO,
            "how-to-guide": cls.HOW_TO,
            "quickstart": cls.QUICKSTART,
            "tutorial": cls.TUTORIAL,
            "concept": cls.CONCEPT,
            "conceptual": cls.CONCEPT,
            "concept-article": cls.CONCEPT,
            "reference": cls.REFERENCE,
            "overview": cls.OVERVIEW,
            "faq": cls.FAQ,
            "troubleshooting": cls.TROUBLESHOOTING,
            "troubleshoot": cls.TROUBLESHOOTING,
            "sample": cls.SAMPLE,
        }

        return mapping.get(ms_topic, cls.UNKNOWN)


class JTBDCoverageStatus(str, Enum):
    """Coverage status for a JTBD step."""

    FULLY_COVERED = "fully_covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    UNKNOWN = "unknown"


class SampleStatus(str, Enum):
    """Status of a code sample."""

    VALID = "valid"
    INVALID = "invalid"
    ORPHANED = "orphaned"
    MISSING = "missing"
    OUTDATED = "outdated"
