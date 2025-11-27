#!/usr/bin/env python3
"""
Smoke Test for RAG Pipeline

Quick validation of all 4 pipeline dimensions before running full 72-variant test.

Tests 13 strategic pipelines covering:
- Phase 1: All 3 extractors (fail-fast)
- Phase 2: All chunking, embedding, retrieval variants
- Phase 3: Critical combinations with recent fixes

Estimated time: ~2 hours
"""
import sys
import os
import json
import argparse
import subprocess
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from test-pipelines.py
from test_pipelines import (
    PipelineConfig,
    PipelineTester,
    EMBEDDING_CONFIGS,
)

# Import validators
from validators import (
    PipelineValidator,
    ValidationResult,
    summarize_validations,
)


# Embedding shortcuts for cleaner code
NOMIC = EMBEDDING_CONFIGS[0]  # nomic-ai/nomic-embed-text-v1.5
OPENAI = EMBEDDING_CONFIGS[1]  # text-embedding-3-small

# Retrieval shortcuts
DENSE = {"strategy": "dense", "reranking": "none"}
HYBRID = {"strategy": "hybrid", "reranking": "none"}
DENSE_RERANK = {"strategy": "dense", "reranking": "cross-encoder"}
HYBRID_RERANK = {"strategy": "hybrid", "reranking": "cross-encoder"}


def define_smoke_tests() -> List[tuple]:
    """
    Define 13 strategic smoke test configurations.

    Returns:
        List of (phase, test_num, description, PipelineConfig)
    """
    tests = []

    # ========== PHASE 1: EXTRACTOR VALIDATION (CRITICAL) ==========
    # Test all 3 extractors with simplest config (LangChain + Nomic + Dense)
    # If these fail, nothing else matters
    tests.extend([
        (1, 1, "Baseline (Docling)",
         PipelineConfig("docling", "langchain", NOMIC, DENSE)),

        (1, 2, "Unstructured extractor",
         PipelineConfig("unstructured", "langchain", NOMIC, DENSE)),

        (1, 3, "LlamaParse extractor",
         PipelineConfig("llamaparse", "langchain", NOMIC, DENSE)),
    ])

    # ========== PHASE 2: DIMENSION COVERAGE (CRITICAL) ==========
    # Test each variant of chunking, embedding, retrieval
    tests.extend([
        (2, 4, "Chonkie chunking",
         PipelineConfig("docling", "chonkie", NOMIC, DENSE)),

        (2, 5, "Semantic chunking",
         PipelineConfig("docling", "semantic", NOMIC, DENSE)),

        (2, 6, "OpenAI embeddings",
         PipelineConfig("docling", "langchain", OPENAI, DENSE)),

        (2, 7, "Hybrid retrieval (RRF fix)",
         PipelineConfig("docling", "langchain", NOMIC, HYBRID)),

        (2, 8, "Dense + Reranking (sigmoid fix)",
         PipelineConfig("docling", "langchain", NOMIC, DENSE_RERANK)),

        (2, 9, "Hybrid + Reranking (both fixes)",
         PipelineConfig("docling", "langchain", NOMIC, HYBRID_RERANK)),
    ])

    # ========== PHASE 3: CRITICAL COMBINATIONS (IMPORTANT) ==========
    # Test recently fixed features with complex configurations
    tests.extend([
        (3, 10, "Current config (all advanced features)",
         PipelineConfig("unstructured", "chonkie", NOMIC, HYBRID_RERANK)),

        (3, 11, "Most advanced pipeline",
         PipelineConfig("llamaparse", "semantic", OPENAI, HYBRID_RERANK)),

        (3, 12, "RRF stress test (complex chunking)",
         PipelineConfig("unstructured", "semantic", NOMIC, HYBRID)),

        (3, 13, "Rerank stress test (quality chunks)",
         PipelineConfig("llamaparse", "chonkie", OPENAI, DENSE_RERANK)),
    ])

    return tests


class SmokeTestRunner:
    """Runs smoke tests with validation and reporting"""

    def __init__(self, results_file: str = "smoke_test_results.json"):
        self.results_file = results_file
        self.report_file = results_file.replace(".json", "_report.txt")
        # Ensure results directory exists
        Path(self.results_file).parent.mkdir(parents=True, exist_ok=True)
        self.tester = PipelineTester(results_file)
        self.validator = PipelineValidator()

        self.test_results = []
        self.validation_results = {}
        self.current_api_embedding = None
        self.current_api_retrieval = None
        self.current_api_reranking = None

        print("\n" + "=" * 70)
        print(" 🧪 RAG PIPELINE SMOKE TEST")
        print("=" * 70)
        print(f" Testing 13 strategic pipelines to validate all dimensions")
        print(f" Estimated time: ~2 hours")
        print(f" Results: {self.results_file}")
        print(f" Report: {self.report_file}")
        print("=" * 70 + "\n")

    def run_all_tests(self, start_from: int = 0, limit: int = None):
        """Run all smoke tests with validation"""
        tests = define_smoke_tests()

        if limit:
            tests = tests[start_from:start_from + limit]
        elif start_from:
            tests = tests[start_from:]

        # ===== START INFRASTRUCTURE ONCE =====
        print("\n⏳ Starting infrastructure (qdrant, postgres, ollama, minio)...")

        subprocess.run(
            ["docker", "compose", "up", "-d", "qdrant", "postgres", "ollama", "minio"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        time.sleep(5)

        print("✓ Infrastructure is ready. rag-api will be started per pipeline config.\n")
        # =============================================

        current_phase = 0
        phase_failures = {1: 0, 2: 0, 3: 0}

        for phase, test_num, description, config in tests:
            # Ensure rag-api is running with the correct config for this pipeline
            self._ensure_api_for_config(config)

            # Print phase header
            if phase != current_phase:
                current_phase = phase
                print(f"\n{'='*70}")
                print(f" PHASE {phase}: {self._get_phase_name(phase)}")
                print(f"{'='*70}\n")

            # Run test
            print(f"\n[Test {test_num}/13] {description}")
            print(f"  Pipeline: {config.pipeline_id}")
            print(f"  Config: {config.extractor} + {config.chunking} + "
                  f"{config.embedding['shortname']} + {config.retrieval['strategy']}"
                  f"{' + reranking' if config.retrieval['reranking'] != 'none' else ''}")

            success, ingestion_result, retrieval_result = self._run_single_test(config)

            # Validate results
            validations = self._validate_test(config, ingestion_result, retrieval_result)
            self.validation_results[config.pipeline_id] = validations

            # Count errors
            _, warnings, errors = summarize_validations(validations)

            if errors > 0:
                phase_failures[phase] += 1
                print(f"\n  ❌ TEST FAILED: {errors} validation errors")
            elif warnings > 0:
                print(f"\n  ⚠️  TEST PASSED WITH WARNINGS: {warnings} warnings")
            else:
                print(f"\n  ✅ TEST PASSED")

            # Print validation results
            for validation in validations:
                if not validation.passed or validation.severity != "info":
                    print(f"    {validation}")

            # Check stop criteria
            if self._should_stop(phase, phase_failures):
                print("\n" + "!" * 70)
                print(" STOP CRITERIA MET - Too many failures detected")
                print("!" * 70)
                break

        # Generate final report
        self._generate_report(tests, phase_failures)

    def _ensure_api_for_config(self, config: PipelineConfig):
        """
        Make sure rag-api is running with the correct embedding, retrieval, and reranking config.
        Restarts rag-api if needed.
        """
        needed_embedding = f"{config.embedding['provider']}:{config.embedding['model']}"
        needed_retrieval = config.retrieval["strategy"]          # "dense" or "hybrid"
        needed_reranking = config.retrieval["reranking"]         # "none" or "cross-encoder"

        # Decide whether we need to restart the API
        restart_needed = (
            needed_embedding != self.current_api_embedding
            or needed_retrieval != self.current_api_retrieval
            or needed_reranking != self.current_api_reranking
        )

        if not restart_needed:
            return  # API already matches this config

        print("\n⏳ Restarting rag-api with new config:")
        print(f"   embedding        = {needed_embedding}")
        print(f"   retrieval        = {needed_retrieval}")
        print(f"   reranking        = {needed_reranking}")

        # Stop old rag-api (if any)
        subprocess.run(
            ["docker", "compose", "rm", "-sf", "rag-api"],
            capture_output=True,
            text=True,
        )

        # Build env for docker compose. This assumes your docker-compose.yml
        # uses these variables in the rag-api service env, e.g.:
        #   environment:
        #     - EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}
        #     - EMBEDDING_MODEL=${EMBEDDING_MODEL}
        #     - RETRIEVAL_STRATEGY=${RETRIEVAL_STRATEGY}
        #     - RERANKING_STRATEGY=${RERANKING_STRATEGY}
        env = os.environ.copy()
        env["EMBEDDING_PROVIDER"] = config.embedding["provider"]
        env["EMBEDDING_MODEL"] = config.embedding["model"]
        env["RETRIEVAL_STRATEGY"] = needed_retrieval
        env["RERANKING_STRATEGY"] = needed_reranking

        # Start rag-api with new env
        subprocess.run(
            ["docker", "compose", "up", "-d", "rag-api"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

        if not self.tester.wait_for_api_ready(max_wait=180):
            raise RuntimeError("rag-api did not become ready after restart")

        # Update current config trackers
        self.current_api_embedding = needed_embedding
        self.current_api_retrieval = needed_retrieval
        self.current_api_reranking = needed_reranking

        # Optional: re-measure baseline memory
        self.tester.measure_api_baseline_memory()

    def _run_single_test(self, config: PipelineConfig) -> tuple:
        """Run a single pipeline test (ingestion + retrieval)"""
        try:
            # Run ingestion
            print(f"\n  [1/2] Running ingestion...")
            ingestion_result = self.tester.run_ingestion(config)

            if ingestion_result["status"] != "success":
                print(f"  ❌ Ingestion failed: {ingestion_result.get('error', 'Unknown error')}")
                return False, ingestion_result, None

            print(f"  ✅ Ingestion completed in {ingestion_result['ingestion_time_seconds']:.1f}s")

            # Run retrieval with only 2 queries (smoke test)
            print(f"\n  [2/2] Running retrieval (2 sample queries)...")
            print(f"  [DEBUG] Using updated quick retrieval method")
            retrieval_result = self._run_quick_retrieval_test(config)

            if retrieval_result["status"] != "success":
                print(f"  ❌ Retrieval failed: {retrieval_result.get('error', 'Unknown error')}")
                return False, ingestion_result, retrieval_result

            print(f"  ✅ Retrieval completed: {retrieval_result['queries_successful']}/{retrieval_result['queries_tested']} queries")

            return True, ingestion_result, retrieval_result

        except Exception as e:
            print(f"  ❌ Test failed with exception: {str(e)}")
            return False, {"status": "error", "error": str(e)}, None

    def _run_quick_retrieval_test(self, config: PipelineConfig) -> Dict:
        """
        Run a quick retrieval test with only 2 queries (smoke test).

        This is much faster than running all 46 queries - we just need to verify
        the API works and returns valid results.
        """
        RAG_API_QUERY = "http://localhost:8080/query"

        # Sample queries (diverse types)
        test_queries = [
            "Welche drei Charakteristika besitzt C?",  # Factual
            "Was bewirkt #include <stdio.h>?",         # Technical
        ]

        try:
            # rag-api is managed by _ensure_api_for_config. Here we only:
            # - override the collection per request
            # - measure retrieval behavior for the current pipeline config

            query_results = []
            successful = 0
            total_score = 0
            total_time = 0

            for i, query in enumerate(test_queries):
                try:
                    start = time.time()
                    response = requests.post(
                        RAG_API_QUERY,
                        json={
                            "query": query,
                            "top_k": 5,
                            "min_score": 0.3,
                            "collection": config.collection_name  # Override collection per test
                        },
                        timeout=60,
                    )
                    elapsed = time.time() - start

                    if response.status_code == 200:
                        data = response.json()
                        sources = data.get("sources", [])
                        avg_score = sum(s["score"] for s in sources) / len(sources) if sources else 0

                        query_results.append({
                            "query": query,
                            "status": "success",
                            "results_count": len(sources),
                            "avg_score": avg_score,
                            "sources": sources,
                        })

                        successful += 1
                        total_score += avg_score
                        total_time += elapsed * 1000

                        print(f"      ✓ Query {i+1}: {len(sources)} results (score: {avg_score:.3f})")
                    else:
                        query_results.append({
                            "query": query,
                            "status": "failed",
                            "error": f"HTTP {response.status_code}",
                        })
                        print(f"      ✗ Query {i+1}: HTTP {response.status_code}")

                except Exception as e:
                    query_results.append({
                        "query": query,
                        "status": "error",
                        "error": str(e),
                    })
                    print(f"      ✗ Query {i+1}: {e}")

            return {
                "status": "success",
                "queries_tested": len(test_queries),
                "queries_successful": successful,
                "avg_score": total_score / successful if successful > 0 else 0,
                "avg_retrieval_time_ms": total_time / successful if successful > 0 else 0,
                "queries": query_results,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "queries_tested": 0,
                "queries_successful": 0,
            }

    def _validate_test(self, config: PipelineConfig, ingestion_result: Dict, retrieval_result: Dict) -> List[ValidationResult]:
        """Run all validations for a test"""
        validations = []

        # Validate ingestion (but skip chunk count check - we only test 1 file)
        if ingestion_result:
            ing_validations = self.validator.validate_ingestion(ingestion_result)
            # Filter out chunk count errors (processed_count checks FILES not chunks)
            validations.extend([v for v in ing_validations if "chunks:" not in v.message])

        # Validate retrieval
        if retrieval_result:
            validations.extend(self.validator.validate_retrieval(retrieval_result, config.pipeline_id))

            # Validate RRF fusion (hybrid only)
            validations.append(self.validator.validate_hybrid_rrf(config.pipeline_id))

            # Validate sigmoid normalization (reranking only)
            validations.append(self.validator.validate_reranking_sigmoid(retrieval_result, config.pipeline_id))

        return validations

    def _should_stop(self, current_phase: int, phase_failures: Dict[int, int]) -> bool:
        """Determine if we should stop testing based on failures"""
        # Stop if >2 Phase 1 tests fail (extractors broken)
        if current_phase == 1 and phase_failures[1] > 2:
            return True

        # Stop if >3 total tests fail (systemic issues)
        total_failures = sum(phase_failures.values())
        if total_failures > 3:
            return True

        return False

    def _get_phase_name(self, phase: int) -> str:
        """Get descriptive name for phase"""
        names = {
            1: "EXTRACTOR VALIDATION (FAIL-FAST)",
            2: "DIMENSION COVERAGE",
            3: "CRITICAL COMBINATIONS"
        }
        return names.get(phase, f"Phase {phase}")

    def _generate_report(self, tests: List[tuple], phase_failures: Dict[int, int]):
        """Generate summary report"""
        print("\n" + "=" * 70)
        print(" SMOKE TEST REPORT")
        print("=" * 70)

        # Count totals
        total_tests = len(self.validation_results)
        total_passed = 0
        total_warnings = 0
        total_errors = 0

        for validations in self.validation_results.values():
            passed, warnings, errors = summarize_validations(validations)
            if errors == 0:
                total_passed += 1
            total_warnings += warnings
            total_errors += errors

        # Print summary
        print(f"\nTests Run: {total_tests}/13")
        print(f"✅ Passed: {total_passed}")
        print(f"⚠️  Warnings: {total_warnings}")
        print(f"❌ Errors: {total_errors}")

        # Phase breakdown
        print(f"\nPhase Failures:")
        for phase in [1, 2, 3]:
            print(f"  Phase {phase}: {phase_failures.get(phase, 0)} failed")

        # Recommendations
        print(f"\n{'=' * 70}")
        print(" RECOMMENDATIONS")
        print("=" * 70)

        if total_errors == 0:
            print("\n✅ All smoke tests passed!")
            print("   You can proceed with full 72-variant evaluation.")
        elif total_errors <= 2:
            print("\n⚠️  Minor issues detected.")
            print("   Review warnings and fix if critical.")
            print("   Consider re-running failed tests before full evaluation.")
        else:
            print("\n❌ Multiple failures detected.")
            print("   DO NOT run full evaluation yet.")
            print("   Debug and fix issues, then re-run smoke test.")

        # Save report to file
        report_path = self.report_file
        with open(report_path, "w") as f:
            f.write("RAG PIPELINE SMOKE TEST REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tests Run: {total_tests}/13\n")
            f.write(f"Passed: {total_passed}\n")
            f.write(f"Warnings: {total_warnings}\n")
            f.write(f"Errors: {total_errors}\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("=" * 70 + "\n\n")

            for pipeline_id, validations in self.validation_results.items():
                f.write(f"\n{pipeline_id}:\n")
                for validation in validations:
                    f.write(f"  {validation}\n")

        print(f"\nDetailed report saved to: {report_path}")
        print("=" * 70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smoke test for RAG pipelines - validates all dimensions before full test"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from test number N (1-13, default: 1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run N tests (default: all remaining)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/smoke_test_results.json",
        help="Output file for results (default: results/smoke_test_results.json)"
    )

    args = parser.parse_args()

    # Adjust start-from to 0-indexed
    start_from = max(0, args.start_from - 1) if args.start_from > 0 else 0

    # Create results directory if needed
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Run smoke tests
    runner = SmokeTestRunner(results_file=args.results_file)
    runner.run_all_tests(start_from=start_from, limit=args.limit)


if __name__ == "__main__":
    main()
