"""Tests for Phase 6 components (JTBD and Samples)."""

import tempfile
from pathlib import Path

import pytest

from foundry_eval.models.code_sample import (
    CodeSample,
    SampleLanguage,
    SampleReference,
    SamplesCoverageReport,
    SampleStatus,
    SampleValidationResult,
)
from foundry_eval.models.jtbd import (
    CoverageGap,
    JTBD,
    JTBDAnalysisResult,
    JTBDCoverageStatus,
    JTBDMapping,
    JTBDStep,
)


class TestJTBDModels:
    """Tests for JTBD models."""

    def test_jtbd_step_creation(self):
        """Test creating a JTBD step."""
        step = JTBDStep(
            step_id="step-1",
            step_number=1,
            title="Install SDK",
            description="Install the Foundry SDK",
        )

        assert step.step_id == "step-1"
        assert step.step_number == 1
        assert step.coverage_status == JTBDCoverageStatus.NOT_COVERED

    def test_jtbd_creation(self):
        """Test creating a JTBD."""
        jtbd = JTBD(
            jtbd_id="jtbd-001",
            title="Build a data pipeline",
            description="Create a data pipeline using Foundry",
            persona="Data Engineer",
            steps=[
                JTBDStep(
                    step_id="step-1",
                    step_number=1,
                    title="Set up environment",
                    description="Set up the development environment",
                ),
                JTBDStep(
                    step_id="step-2",
                    step_number=2,
                    title="Configure data sources",
                    description="Configure input data sources",
                ),
            ],
        )

        assert jtbd.jtbd_id == "jtbd-001"
        assert jtbd.total_steps == 2
        assert jtbd.covered_steps == 0
        assert jtbd.coverage_percentage == 0.0

    def test_jtbd_coverage_calculation(self):
        """Test JTBD coverage calculation."""
        jtbd = JTBD(
            jtbd_id="jtbd-001",
            title="Test JTBD",
            description="Test",
            steps=[
                JTBDStep(
                    step_id="step-1",
                    step_number=1,
                    title="Step 1",
                    description="Description",
                    coverage_status=JTBDCoverageStatus.FULLY_COVERED,
                ),
                JTBDStep(
                    step_id="step-2",
                    step_number=2,
                    title="Step 2",
                    description="Description",
                    coverage_status=JTBDCoverageStatus.NOT_COVERED,
                ),
            ],
        )

        assert jtbd.covered_steps == 1
        assert jtbd.coverage_percentage == 50.0

    def test_coverage_gap_creation(self):
        """Test creating a coverage gap."""
        gap = CoverageGap(
            jtbd_id="jtbd-001",
            step_id="step-2",
            gap_type="missing",
            severity="high",
            title="Missing configuration docs",
            description="No documentation for configuration step",
            suggested_article_title="How to Configure Data Sources",
        )

        assert gap.gap_type == "missing"
        assert gap.severity == "high"

    def test_jtbd_analysis_result(self):
        """Test JTBD analysis result."""
        jtbd = JTBD(
            jtbd_id="jtbd-001",
            title="Test JTBD",
            description="Test",
            steps=[],
        )

        result = JTBDAnalysisResult(
            jtbd=jtbd,
            gaps=[
                CoverageGap(
                    jtbd_id="jtbd-001",
                    step_id="step-1",
                    gap_type="missing",
                    severity="critical",
                    title="Critical gap",
                    description="Critical gap",
                ),
                CoverageGap(
                    jtbd_id="jtbd-001",
                    step_id="step-2",
                    gap_type="incomplete",
                    severity="high",
                    title="High gap",
                    description="High gap",
                ),
            ],
        )

        assert result.has_critical_gaps is True
        assert result.gap_count_by_severity["critical"] == 1
        assert result.gap_count_by_severity["high"] == 1


class TestCodeSampleModels:
    """Tests for code sample models."""

    def test_code_sample_creation(self):
        """Test creating a code sample."""
        sample = CodeSample(
            sample_id="samples/auth/basic.py",
            file_path="samples/auth/basic.py",
            language=SampleLanguage.PYTHON,
            title="Basic Authentication",
            source_file="/path/to/basic.py",
            content="def authenticate():\n    pass",
        )

        assert sample.sample_id == "samples/auth/basic.py"
        assert sample.language == SampleLanguage.PYTHON
        assert sample.is_orphaned is True  # No references
        assert sample.line_count == 2

    def test_sample_reference_creation(self):
        """Test creating a sample reference."""
        ref = SampleReference(
            article_path="docs/auth.md",
            sample_id="samples/auth/basic.py",
            reference_type="include",
            is_valid=True,
        )

        assert ref.is_valid is True
        assert ref.reference_type == "include"

    def test_sample_validation_result(self):
        """Test sample validation result."""
        sample = CodeSample(
            sample_id="test.py",
            file_path="test.py",
            language=SampleLanguage.PYTHON,
            source_file="/path/test.py",
            content="print('hello')",
        )

        result = SampleValidationResult(
            sample=sample,
            status=SampleStatus.VALID,
            completeness_score=0.8,
            correctness_score=0.9,
            best_practices_score=0.7,
        )

        assert result.is_valid is True
        # 0.8 * 0.4 + 0.9 * 0.4 + 0.7 * 0.2 = 0.32 + 0.36 + 0.14 = 0.82
        assert result.overall_score == pytest.approx(0.82)

    def test_samples_coverage_report(self):
        """Test samples coverage report."""
        report = SamplesCoverageReport(
            total_samples=100,
            valid_samples=85,
            invalid_samples=15,
            orphaned_samples=10,
            total_articles=50,
            articles_with_samples=40,
            articles_needing_samples=45,
            articles_missing_samples=["doc1.md", "doc2.md", "doc3.md", "doc4.md", "doc5.md"],
        )

        assert report.validation_pass_rate == 85.0
        # (45 - 5) / 45 * 100 = 88.89%
        assert report.sample_coverage_percentage == pytest.approx(88.89, rel=0.01)


class TestJTBDLoader:
    """Tests for JTBD loader."""

    @pytest.fixture
    def sample_jtbd_json(self, tmp_path):
        """Create a sample JTBD JSON file."""
        import json

        data = {
            "jtbds": [
                {
                    "jtbd_id": "jtbd-001",
                    "title": "Build Data Pipeline",
                    "description": "Create a data pipeline",
                    "persona": "Data Engineer",
                    "steps": [
                        {
                            "step_id": "step-1",
                            "step_number": 1,
                            "title": "Setup",
                            "description": "Set up environment",
                        },
                        {
                            "step_id": "step-2",
                            "step_number": 2,
                            "title": "Configure",
                            "description": "Configure sources",
                        },
                    ],
                }
            ]
        }

        file_path = tmp_path / "jtbd.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    @pytest.mark.asyncio
    async def test_load_json(self, sample_jtbd_json):
        """Test loading JTBD from JSON."""
        from foundry_eval.context.jtbd_loader import JTBDLoader

        loader = JTBDLoader(sample_jtbd_json)
        jtbds = await loader.load_all()

        assert len(jtbds) == 1
        assert jtbds[0].jtbd_id == "jtbd-001"
        assert len(jtbds[0].steps) == 2

    @pytest.mark.asyncio
    async def test_load_by_id(self, sample_jtbd_json):
        """Test loading specific JTBD by ID."""
        from foundry_eval.context.jtbd_loader import JTBDLoader

        loader = JTBDLoader(sample_jtbd_json)
        jtbd = await loader.load_by_id("jtbd-001")

        assert jtbd is not None
        assert jtbd.title == "Build Data Pipeline"

    @pytest.mark.asyncio
    async def test_load_nonexistent_id(self, sample_jtbd_json):
        """Test loading nonexistent JTBD ID."""
        from foundry_eval.context.jtbd_loader import JTBDLoader

        loader = JTBDLoader(sample_jtbd_json)
        jtbd = await loader.load_by_id("nonexistent")

        assert jtbd is None

    @pytest.fixture
    def sample_golden_path_csv(self, tmp_path):
        """Create a sample Golden Path format CSV file."""
        data = """JTBD #,Title,URL,Phase,Area,Assignees,JTBD Status,Target Grade
1,Create resource group for developer sandbox,https://github.com/example/1,Idea to Proto,Provision,Alice,Writing in Progress,B
2,Set up data pipeline with transforms,https://github.com/example/2,Build to Deploy,Data & Pipeline,"Bob, Charlie",Published,A
3,Configure role-based access control,https://github.com/example/3,Build to Deploy,Security,,Not Started,C
"""
        file_path = tmp_path / "golden_path.csv"
        file_path.write_text(data)
        return file_path

    @pytest.mark.asyncio
    async def test_load_golden_path_csv(self, sample_golden_path_csv):
        """Test loading JTBD from Golden Path CSV format."""
        from foundry_eval.context.jtbd_loader import JTBDLoader

        loader = JTBDLoader(sample_golden_path_csv)
        jtbds = await loader.load_all()

        assert len(jtbds) == 3

        # Check first JTBD
        jtbd1 = jtbds[0]
        assert jtbd1.jtbd_id == "jtbd-1"
        assert jtbd1.title == "Create resource group for developer sandbox"
        assert jtbd1.phase == "Idea to Proto"
        assert jtbd1.area == "Provision"
        assert jtbd1.status == "Writing in Progress"
        assert jtbd1.is_stepless is True
        assert len(jtbd1.steps) == 0

        # Check persona derivation
        assert jtbd1.persona == "Platform Admin"  # Derived from "Provision"

        # Check second JTBD with multiple assignees
        jtbd2 = jtbds[1]
        assert jtbd2.jtbd_id == "jtbd-2"
        assert len(jtbd2.assignees) == 2
        assert "Bob" in jtbd2.assignees
        assert "Charlie" in jtbd2.assignees
        assert jtbd2.persona == "Data Engineer"  # Derived from "Data & Pipeline"

    @pytest.mark.asyncio
    async def test_stepless_jtbd_coverage(self, sample_golden_path_csv):
        """Test coverage properties for step-less JTBDs."""
        from foundry_eval.context.jtbd_loader import JTBDLoader
        from foundry_eval.models.jtbd import JTBDCoverageStatus

        loader = JTBDLoader(sample_golden_path_csv)
        jtbds = await loader.load_all()
        jtbd = jtbds[0]

        # Initial state
        assert jtbd.coverage_percentage == 0.0
        assert jtbd.coverage_status == JTBDCoverageStatus.NOT_COVERED

        # Update coverage
        jtbd.coverage_status = JTBDCoverageStatus.PARTIALLY_COVERED
        jtbd.coverage_score = 0.65
        assert jtbd.coverage_percentage == 65.0


class TestSamplesIndex:
    """Tests for samples index."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository structure."""
        # Create sample files
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()

        (samples_dir / "auth.py").write_text('"""Authentication example."""\nprint("auth")')
        (samples_dir / "data.py").write_text("# Data processing\nprint('data')")
        (samples_dir / "config.json").write_text('{"key": "value"}')

        return samples_dir

    @pytest.mark.asyncio
    async def test_index_samples(self, sample_repo):
        """Test indexing samples."""
        from foundry_eval.context.samples_index import SamplesIndex

        index = SamplesIndex(sample_repo)
        count = await index.index()

        assert count == 3
        assert len(index.samples) == 3

    @pytest.mark.asyncio
    async def test_get_sample(self, sample_repo):
        """Test getting a sample by ID."""
        from foundry_eval.context.samples_index import SamplesIndex

        index = SamplesIndex(sample_repo)
        await index.index()

        sample = await index.get_sample("auth.py")

        assert sample is not None
        assert sample.language == SampleLanguage.PYTHON

    @pytest.mark.asyncio
    async def test_orphaned_samples(self, sample_repo):
        """Test getting orphaned samples."""
        from foundry_eval.context.samples_index import SamplesIndex

        index = SamplesIndex(sample_repo)
        await index.index()

        orphaned = await index.get_orphaned_samples()

        # All samples are orphaned since no articles reference them
        assert len(orphaned) == 3

    @pytest.mark.asyncio
    async def test_samples_by_language(self, sample_repo):
        """Test getting samples by language."""
        from foundry_eval.context.samples_index import SamplesIndex

        index = SamplesIndex(sample_repo)
        await index.index()

        python_samples = await index.get_samples_by_language(SampleLanguage.PYTHON)

        assert len(python_samples) == 2  # auth.py and data.py
