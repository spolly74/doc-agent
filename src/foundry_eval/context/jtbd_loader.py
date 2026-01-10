"""JTBD data loader for CSV and JSON formats."""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from foundry_eval.models.jtbd import JTBD, JTBDStep

logger = logging.getLogger("foundry_eval.context.jtbd_loader")


class JTBDLoader:
    """Loads JTBD definitions from CSV or JSON files."""

    def __init__(self, jtbd_path: Path):
        """Initialize the JTBD loader.

        Args:
            jtbd_path: Path to JTBD data file (CSV or JSON).
        """
        self._path = jtbd_path

    async def load_all(self) -> list[JTBD]:
        """Load all JTBD definitions from the file.

        Returns:
            List of JTBD objects.

        Raises:
            ValueError: If the file format is not supported.
        """
        suffix = self._path.suffix.lower()

        if suffix == ".json":
            return await self._load_json()
        elif suffix == ".csv":
            return await self._load_csv()
        else:
            raise ValueError(f"Unsupported JTBD file format: {suffix}")

    async def load_by_id(self, jtbd_id: str) -> Optional[JTBD]:
        """Load a specific JTBD by ID.

        Args:
            jtbd_id: The JTBD ID to load.

        Returns:
            JTBD object if found, None otherwise.
        """
        all_jtbds = await self.load_all()
        for jtbd in all_jtbds:
            if jtbd.jtbd_id == jtbd_id:
                return jtbd
        return None

    async def _load_json(self) -> list[JTBD]:
        """Load JTBD definitions from JSON file.

        Expected JSON structure:
        {
            "jtbds": [
                {
                    "jtbd_id": "jtbd-001",
                    "title": "...",
                    "description": "...",
                    "persona": "...",
                    "steps": [
                        {
                            "step_id": "step-1",
                            "step_number": 1,
                            "title": "...",
                            "description": "..."
                        }
                    ]
                }
            ]
        }
        """
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        jtbds = []

        # Handle both array and object with "jtbds" key
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "jtbds" in data:
            items = data["jtbds"]
        else:
            items = [data]  # Single JTBD object

        for item in items:
            jtbd = self._parse_jtbd_dict(item)
            if jtbd:
                jtbds.append(jtbd)

        logger.info(f"Loaded {len(jtbds)} JTBD definitions from {self._path}")
        return jtbds

    async def _load_csv(self) -> list[JTBD]:
        """Load JTBD definitions from CSV file.

        Expected CSV columns:
        - jtbd_id, title, description, persona, category, priority
        - step_id, step_number, step_title, step_description

        Each row represents a step, with JTBD metadata repeated.
        """
        jtbds_dict: dict[str, dict] = {}

        with open(self._path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                jtbd_id = row.get("jtbd_id", "").strip()
                if not jtbd_id:
                    continue

                # Initialize JTBD if not seen before
                if jtbd_id not in jtbds_dict:
                    jtbds_dict[jtbd_id] = {
                        "jtbd_id": jtbd_id,
                        "title": row.get("title", "").strip(),
                        "description": row.get("description", "").strip(),
                        "persona": row.get("persona", "").strip(),
                        "category": row.get("category", "").strip(),
                        "priority": int(row.get("priority", 0) or 0),
                        "steps": [],
                    }

                # Add step if present
                step_id = row.get("step_id", "").strip()
                if step_id:
                    step = {
                        "step_id": step_id,
                        "step_number": int(row.get("step_number", 0) or 0),
                        "title": row.get("step_title", "").strip(),
                        "description": row.get("step_description", "").strip(),
                    }
                    jtbds_dict[jtbd_id]["steps"].append(step)

        # Convert to JTBD objects
        jtbds = []
        for jtbd_data in jtbds_dict.values():
            jtbd = self._parse_jtbd_dict(jtbd_data)
            if jtbd:
                jtbds.append(jtbd)

        logger.info(f"Loaded {len(jtbds)} JTBD definitions from {self._path}")
        return jtbds

    def _parse_jtbd_dict(self, data: dict) -> Optional[JTBD]:
        """Parse a dictionary into a JTBD object.

        Args:
            data: Dictionary with JTBD data.

        Returns:
            JTBD object or None if parsing fails.
        """
        try:
            # Parse steps
            steps = []
            for step_data in data.get("steps", []):
                step = JTBDStep(
                    step_id=step_data.get("step_id", ""),
                    step_number=int(step_data.get("step_number", 0)),
                    title=step_data.get("title", step_data.get("step_title", "")),
                    description=step_data.get("description", step_data.get("step_description", "")),
                    required_skills=step_data.get("required_skills", []),
                    expected_outcomes=step_data.get("expected_outcomes", []),
                )
                steps.append(step)

            # Sort steps by step_number
            steps.sort(key=lambda s: s.step_number)

            # Parse prerequisites and tags
            prerequisites = data.get("prerequisites", [])
            if isinstance(prerequisites, str):
                prerequisites = [p.strip() for p in prerequisites.split(",") if p.strip()]

            tags = data.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            return JTBD(
                jtbd_id=data.get("jtbd_id", ""),
                title=data.get("title", ""),
                description=data.get("description", ""),
                persona=data.get("persona", ""),
                category=data.get("category", ""),
                priority=int(data.get("priority", 0) or 0),
                steps=steps,
                prerequisites=prerequisites,
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Failed to parse JTBD: {e}")
            return None


def get_jtbd_loader(jtbd_path: Path) -> JTBDLoader:
    """Factory function to create a JTBD loader.

    Args:
        jtbd_path: Path to JTBD data file.

    Returns:
        JTBDLoader instance.
    """
    return JTBDLoader(jtbd_path)
