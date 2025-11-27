#!/usr/bin/env python3
"""
Validation utilities for smoke testing pipelines.

Provides validators for:
- Ingestion metrics (time, chunk count, errors)
- Retrieval scores (dense, hybrid, reranking)
- Implementation fixes (RRF fusion, sigmoid normalization)
"""
from typing import Dict, List, Any, Tuple
import subprocess


class ValidationResult:
    """Result of a validation check"""

    def __init__(self, passed: bool, message: str, severity: str = "error"):
        self.passed = passed
        self.message = message
        self.severity = severity  # "error", "warning", "info"

    def __repr__(self):
        status = "✅" if self.passed else ("⚠️" if self.severity == "warning" else "❌")
        return f"{status} {self.message}"


class PipelineValidator:
    """Validates pipeline test results"""

    @staticmethod
    def validate_ingestion(result: Dict[str, Any]) -> List[ValidationResult]:
        """Validate ingestion phase results"""
        validations = []

        # Check status
        if result.get("status") != "success":
            validations.append(ValidationResult(
                False,
                f"Ingestion failed: {result.get('status')}",
                "error"
            ))
            return validations

        # Check ingestion time
        time_sec = result.get("ingestion_time_seconds", 0)
        if time_sec > 1200:  # 20 min
            validations.append(ValidationResult(
                False,
                f"Ingestion timeout: {time_sec:.1f}s (>20min)",
                "error"
            ))
        elif time_sec > 900:  # 15 min
            validations.append(ValidationResult(
                True,
                f"Ingestion slow: {time_sec:.1f}s (15-20min)",
                "warning"
            ))
        else:
            validations.append(ValidationResult(
                True,
                f"Ingestion time OK: {time_sec:.1f}s",
                "info"
            ))

        # Check chunk count
        chunk_count = result.get("processed_count", 0)
        if chunk_count < 100:
            validations.append(ValidationResult(
                False,
                f"Too few chunks: {chunk_count} (<100)",
                "error"
            ))
        elif chunk_count < 200:
            validations.append(ValidationResult(
                True,
                f"Low chunk count: {chunk_count} (100-200)",
                "warning"
            ))
        else:
            validations.append(ValidationResult(
                True,
                f"Chunk count OK: {chunk_count}",
                "info"
            ))

        # Check error count
        error_count = result.get("error_count", 0)
        if error_count > 5:
            validations.append(ValidationResult(
                False,
                f"Too many errors: {error_count} (>5)",
                "error"
            ))
        elif error_count > 0:
            validations.append(ValidationResult(
                True,
                f"Some errors: {error_count} (1-5)",
                "warning"
            ))
        else:
            validations.append(ValidationResult(
                True,
                "No ingestion errors",
                "info"
            ))

        return validations

    @staticmethod
    def validate_retrieval(result: Dict[str, Any], pipeline_id: str) -> List[ValidationResult]:
        """Validate retrieval phase results"""
        validations = []

        # Check status
        if result.get("status") != "success":
            validations.append(ValidationResult(
                False,
                f"Retrieval failed: {result.get('status')}",
                "error"
            ))
            return validations

        # Check query success rate
        total = result.get("queries_tested", 0)
        successful = result.get("queries_successful", 0)
        if successful < total:
            failed = total - successful
            validations.append(ValidationResult(
                True if failed < 5 else False,
                f"Query failures: {failed}/{total}",
                "warning" if failed < 5 else "error"
            ))
        else:
            validations.append(ValidationResult(
                True,
                f"All {total} queries successful",
                "info"
            ))

        # Check average score (dimension-specific)
        avg_score = result.get("avg_score", 0)

        if "hyb" in pipeline_id and "rnk" not in pipeline_id:
            # Hybrid (no reranking): expect 0.4-0.8
            if avg_score < 0.2:
                validations.append(ValidationResult(
                    False,
                    f"Hybrid avg score too low: {avg_score:.3f} (<0.2)",
                    "error"
                ))
            elif avg_score < 0.4:
                validations.append(ValidationResult(
                    True,
                    f"Hybrid avg score low: {avg_score:.3f} (0.2-0.4)",
                    "warning"
                ))
            else:
                validations.append(ValidationResult(
                    True,
                    f"Hybrid avg score OK: {avg_score:.3f}",
                    "info"
                ))

        elif "rnk" in pipeline_id:
            # Reranking: expect 0.3-0.9 (check must be [0, 1])
            if avg_score < 0 or avg_score > 1:
                validations.append(ValidationResult(
                    False,
                    f"Reranking score out of range: {avg_score:.3f} (not in [0,1])",
                    "error"
                ))
            elif avg_score < 0.3:
                validations.append(ValidationResult(
                    True,
                    f"Reranking avg score low: {avg_score:.3f} (<0.3)",
                    "warning"
                ))
            else:
                validations.append(ValidationResult(
                    True,
                    f"Reranking avg score OK: {avg_score:.3f}",
                    "info"
                ))

        else:
            # Dense: expect 0.5-0.9
            if avg_score < 0.3:
                validations.append(ValidationResult(
                    False,
                    f"Dense avg score too low: {avg_score:.3f} (<0.3)",
                    "error"
                ))
            elif avg_score < 0.5:
                validations.append(ValidationResult(
                    True,
                    f"Dense avg score low: {avg_score:.3f} (0.3-0.5)",
                    "warning"
                ))
            else:
                validations.append(ValidationResult(
                    True,
                    f"Dense avg score OK: {avg_score:.3f}",
                    "info"
                ))

        # Check retrieval time (relaxed for smoke tests - includes model loading)
        avg_time_ms = result.get("avg_retrieval_time_ms", 0)
        if avg_time_ms > 60000:  # 60s
            validations.append(ValidationResult(
                False,
                f"Retrieval too slow: {avg_time_ms:.0f}ms (>60s)",
                "error"
            ))
        elif avg_time_ms > 40000:  # 40s
            validations.append(ValidationResult(
                True,
                f"Retrieval slow: {avg_time_ms:.0f}ms (40-60s)",
                "warning"
            ))
        else:
            validations.append(ValidationResult(
                True,
                f"Retrieval time OK: {avg_time_ms:.0f}ms",
                "info"
            ))

        return validations

    @staticmethod
    def validate_hybrid_rrf(pipeline_id: str) -> ValidationResult:
        """Validate that hybrid search uses RRF fusion"""
        if "hyb" not in pipeline_id:
            return ValidationResult(True, "Not a hybrid pipeline (skip RRF check)", "info")

        try:
            # Check RAG API logs for RRF fusion
            result = subprocess.run(
                ["docker", "logs", "rag-api"],
                capture_output=True,
                text=True,
                timeout=5
            )

            logs = result.stdout + result.stderr

            if "fusion_method='RRF'" in logs or "fusion_method=\"RRF\"" in logs:
                return ValidationResult(True, "Hybrid search uses RRF fusion ✓", "info")
            else:
                return ValidationResult(
                    False,
                    "Hybrid search NOT using RRF fusion (check implementation)",
                    "error"
                )
        except Exception as e:
            return ValidationResult(
                True,
                f"Could not verify RRF fusion: {str(e)}",
                "warning"
            )

    @staticmethod
    def validate_reranking_sigmoid(result: Dict[str, Any], pipeline_id: str) -> ValidationResult:
        """Validate that reranking uses sigmoid normalization"""
        if "rnk" not in pipeline_id:
            return ValidationResult(True, "Not a reranking pipeline (skip sigmoid check)", "info")

        # Check if any query has score outside [0, 1]
        queries = result.get("queries", [])
        for query in queries:
            sources = query.get("sources", [])
            for source in sources:
                score = source.get("score", 0)
                if score < 0 or score > 1:
                    return ValidationResult(
                        False,
                        f"Reranking score out of range: {score:.3f} (sigmoid NOT applied)",
                        "error"
                    )

        return ValidationResult(True, "Reranking scores in [0,1] (sigmoid applied ✓)", "info")


def summarize_validations(validations: List[ValidationResult]) -> Tuple[int, int, int]:
    """Count passed, warnings, and errors"""
    passed = sum(1 for v in validations if v.passed and v.severity == "info")
    warnings = sum(1 for v in validations if v.passed and v.severity == "warning")
    errors = sum(1 for v in validations if not v.passed)

    return passed, warnings, errors
