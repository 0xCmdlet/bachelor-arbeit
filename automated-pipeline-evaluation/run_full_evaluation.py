#!/usr/bin/env python3
"""
Pipeline Testing Automation

Systematically tests all combinations of:
- Extraction strategies (docling, unstructured, llamaparse)
- Chunking strategies (langchain, chonkie, semantic, native)
- Embedding providers (sentence-transformers, openai)
- Retrieval strategies (dense, hybrid, cross-encoder, hybrid-reranking)

Generates unique Qdrant collection names for each combination and runs
ingestion + retrieval tests.
"""
import sys
import json
import subprocess
import time
import requests
from typing import List, Dict, Any, Optional
from itertools import product
from datetime import datetime
from pathlib import Path
import os  # <-- needed for passing env to docker compose

# Add current directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent))

from env_updater import EnvUpdater
from log_monitor import LogMonitor


# Define all pipeline variants
EXTRACTORS = [
    "docling",
    "unstructured",
    "llamaparse",
]

CHUNKING_STRATEGIES = [
    "langchain",
    "chonkie",
    "semantic",
    "native",  # For extractors that handle chunking themselves (e.g., Unstructured)
]

EMBEDDING_CONFIGS = [
    {
        "provider": "sentence-transformers",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "shortname": "nomic",
    },
    {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "shortname": "openai3small",
    },
]

RETRIEVAL_STRATEGIES = [
    {"strategy": "dense", "reranking": "none"},
    {"strategy": "hybrid", "reranking": "none"},
    {"strategy": "dense", "reranking": "cross-encoder"},
    {"strategy": "hybrid", "reranking": "cross-encoder"},
]

# Test configuration
RAG_API_URL = "http://localhost:8080"
RAG_API_HEALTH = f"{RAG_API_URL}/health"
RAG_API_QUERY = f"{RAG_API_URL}/query"
RAG_API_STATS = f"{RAG_API_URL}/stats"

# Timeouts
INGESTION_TIMEOUT = 1200  # 20 minutes
API_STARTUP_WAIT = 15  # 15 seconds
API_HEALTH_TIMEOUT = 120  # 2 minutes (allows time for embedding + reranker models to load)


class PipelineConfig:
    """Represents a single pipeline configuration"""

    def __init__(
        self,
        extractor: str,
        chunking: str,
        embedding: Dict[str, str],
        retrieval: Dict[str, str],
    ):
        self.extractor = extractor
        self.chunking = chunking
        self.embedding = embedding
        self.retrieval = retrieval

        # Generate unique collection name
        self.collection_name = self._generate_collection_name()

        # Generate unique pipeline ID
        self.pipeline_id = self._generate_pipeline_id()

    def _generate_collection_name(self) -> str:
        """Generate Qdrant collection name for this pipeline"""
        parts = [
            "docs",
            self.extractor,
            self.chunking,
            self.embedding["shortname"],
            self.retrieval["strategy"],
        ]
        if self.retrieval["reranking"] != "none":
            parts.append("rerank")

        return "_".join(parts)

    def _generate_pipeline_id(self) -> str:
        """Generate short pipeline ID"""
        parts = [
            self.extractor[:3],  # doc/uns/lla
            self.chunking[:3],  # lan/cho/sem
            self.embedding["shortname"][:4],  # nomi/open
            self.retrieval["strategy"][:3],  # den/hyb
        ]
        if self.retrieval["reranking"] != "none":
            parts.append("rnk")

        return "_".join(parts)

    def to_env_dict(self) -> Dict[str, str]:
        """Convert to environment variables dict"""
        env = {
            # Extraction
            "EXTRACTOR_TYPE": self.extractor,
            # Chunking
            "CHUNKING_STRATEGY": self.chunking,
            # Embedding
            "EMBEDDING_PROVIDER": self.embedding["provider"],
            "EMBEDDING_MODEL": self.embedding.get(
                "model", "nomic-ai/nomic-embed-text-v1.5"
            ),
            "OPENAI_EMBEDDING_MODEL": self.embedding.get(
                "model", "text-embedding-3-small"
            )
            if self.embedding["provider"] == "openai"
            else "text-embedding-3-small",
            # Retrieval
            "RETRIEVAL_STRATEGY": self.retrieval["strategy"],
            "RERANKING_STRATEGY": self.retrieval["reranking"],
            # Collection
            "QDRANT_COLLECTION": self.collection_name,
        }
        return env

    def __repr__(self) -> str:
        return f"Pipeline({self.pipeline_id})"


class PipelineTester:
    """Orchestrates pipeline testing"""

    def __init__(self, results_file: str = "pipeline_test_results.json"):
        self.results_file = results_file
        self.results = []
        self.env_updater = EnvUpdater()
        self.log_monitor = LogMonitor(container_name="worker", timeout=INGESTION_TIMEOUT)

        # Load test queries
        self.test_queries = self._load_test_queries()

        # Track current API state for smart lifecycle management
        self.current_api_embedding = None   # "provider:model"
        self.current_api_retrieval = None   # "dense" / "hybrid"
        self.current_api_reranking = None   # "none" / "cross-encoder"
        self.api_baseline_memory_mb = None  # GPU memory baseline with API running

    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from config file"""
        try:
            with open("config/test-queries.json", "r") as f:
                data = json.load(f)
                return data["queries"]
        except FileNotFoundError:
            print(f"❌ Error: config/test-queries.json not found")
            print(f"   Please ensure the test queries config file exists")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in config/test-queries.json")
            print(f"   {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading test queries: {e}")
            raise

    def check_gpu_memory_free(self) -> Optional[tuple]:
        """
        Check GPU memory using nvidia-smi.

        Returns:
            (free_mb, total_mb) or None if nvidia-smi not available
        """
        try:
            # Get free memory
            result_free = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Get total memory
            result_total = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result_free.returncode == 0 and result_total.returncode == 0:
                # Parse first GPU (head -n1)
                free_mb = int(result_free.stdout.strip().split('\n')[0])
                total_mb = int(result_total.stdout.strip().split('\n')[0])
                return (free_mb, total_mb)

            return None

        except Exception as e:
            print(f"   ⚠️  Could not query GPU memory: {e}")
            return None

    def wait_for_gpu_memory_clear(self, threshold_percent: float = 90.0, max_wait: int = 60):
        """
        Wait until GPU memory returns to acceptable level.

        For 24GB VRAM with API running:
        - Uses threshold_percent of FREE memory (not total)
        - 50% threshold = allow ~12GB used (API baseline ~2-3GB + margin)
        - 90% threshold = aggressive cleanup (API must be stopped)

        Args:
            threshold_percent: Percentage of total memory that must be free (default: 90%)
            max_wait: Maximum time to wait in seconds (default: 60s)
        """
        print(f"   Waiting for GPU memory to clear (>{threshold_percent}% free)...")

        start_time = time.time()
        last_log_time = start_time

        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > max_wait:
                print(f"   ⚠️  GPU memory did not clear within {max_wait}s, continuing anyway...")
                return

            # Check GPU memory
            result = self.check_gpu_memory_free()

            if result is None:
                # nvidia-smi not available, skip check
                print(f"   ⚠️  nvidia-smi not available, skipping GPU memory check")
                return

            free_mb, total_mb = result
            used_mb = total_mb - free_mb
            percent_free = (free_mb / total_mb) * 100

            # Log progress every 5 seconds with baseline context
            if time.time() - last_log_time > 5:
                if self.api_baseline_memory_mb:
                    baseline_str = f" (baseline: {self.api_baseline_memory_mb} MiB)"
                else:
                    baseline_str = ""
                print(f"   GPU: {used_mb}/{total_mb} MiB used ({percent_free:.1f}% free){baseline_str}")
                last_log_time = time.time()

            # Check if threshold met
            if percent_free >= threshold_percent:
                print(f"   ✓ GPU memory clear: {free_mb}/{total_mb} MiB free ({percent_free:.1f}%)")
                return

            # Wait before next check
            time.sleep(2)

    def generate_all_pipelines(self) -> List[PipelineConfig]:
        """Generate all valid pipeline combinations (with compatibility filtering)"""
        pipelines = []

        for extractor, chunking, embedding, retrieval in product(
            EXTRACTORS, CHUNKING_STRATEGIES, EMBEDDING_CONFIGS, RETRIEVAL_STRATEGIES
        ):
            # Filter incompatible combinations:
            # 1. Unstructured API already chunks content - must use 'native' chunking
            if extractor == "unstructured" and chunking != "native":
                continue  # Skip: Unstructured only works with native chunking

            # 2. 'native' chunking is only for extractors that handle chunking themselves
            if extractor != "unstructured" and chunking == "native":
                continue  # Skip: native chunking only for Unstructured

            pipeline = PipelineConfig(
                extractor=extractor,
                chunking=chunking,
                embedding=embedding,
                retrieval=retrieval,
            )
            pipelines.append(pipeline)

        return pipelines

    def run_ingestion(self, pipeline: PipelineConfig) -> Dict[str, Any]:
        """
        Run ingestion for a pipeline configuration.

        Returns:
            Result dict with status, timing, and metrics
        """
        print(f"\n{'='*80}")
        print(f"INGESTION: {pipeline.pipeline_id}")
        print(f"Collection: {pipeline.collection_name}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            # Step 1: Update .env file
            print("\n1️⃣  Updating .env file...")
            env_updates = pipeline.to_env_dict()
            success = self.env_updater.update(env_updates, create_backup=True)

            if not success:
                raise Exception("Failed to update .env file")

            print(f"   ✓ Updated {len(env_updates)} environment variables")

            # Step 2: Rebuild worker container
            print("\n2️⃣  Rebuilding worker container...")
            result = subprocess.run(
                ["docker", "compose", "build", "worker"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute build timeout
            )

            if result.returncode != 0:
                raise Exception(f"Build failed: {result.stderr}")

            print("   ✓ Worker container built successfully")

            # Step 3: Start worker
            print("\n3️⃣  Starting worker...")
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "worker"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise Exception(f"Start failed: {result.stderr}")

            print("   ✓ Worker container started")

            # Step 4: Monitor logs for completion
            print("\n4️⃣  Monitoring ingestion progress...")
            monitor_result = self.log_monitor.wait_for_completion()

            if monitor_result["status"] != "success":
                raise Exception(f"Ingestion did not complete successfully: {monitor_result['status']}")

            elapsed = time.time() - start_time

            result = {
                "pipeline_id": pipeline.pipeline_id,
                "collection": pipeline.collection_name,
                "config": {
                    "extractor": pipeline.extractor,
                    "chunking": pipeline.chunking,
                    "embedding": pipeline.embedding["provider"],
                    "embedding_model": pipeline.embedding["model"],
                    "retrieval": pipeline.retrieval["strategy"],
                    "reranking": pipeline.retrieval["reranking"],
                },
                "status": "success",
                "ingestion_time_seconds": elapsed,
                "processed_count": monitor_result["processed_count"],
                "error_count": monitor_result["error_count"],
                "timestamp": datetime.now().isoformat(),
            }

            print(f"\n✅ Ingestion completed successfully")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Processed: {monitor_result['processed_count']} documents")
            print(f"   Errors: {monitor_result['error_count']}")

            # Step 5: GPU memory cleanup (stop worker only, keep infrastructure running)            
            print("\n5️⃣  Cleaning up worker (keeping API and infrastructure running)...")

            cleanup_result = subprocess.run(
                ["docker", "compose", "stop", "worker"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if cleanup_result.returncode == 0:
                print("   ✓ Worker stopped")
            else:
                print("   ⚠️  Warning: Worker stop had issues")
                # Fall back to force removal if stop fails
                subprocess.run(
                    ["docker", "compose", "rm", "-sf", "worker"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                print("   ✓ Worker force-removed (fallback)")

            # Poll for GPU memory clearance instead of arbitrary sleep
            print("   Waiting for GPU memory to clear...")
            self.wait_for_gpu_memory_clear(threshold_percent=50.0, max_wait=120)

            return result

        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Ingestion timed out after {elapsed:.1f}s")

            # Clean up worker only (keep API and infrastructure running)
            subprocess.run(["docker", "compose", "stop", "worker"], capture_output=True, timeout=30)
            self.wait_for_gpu_memory_clear(threshold_percent=50.0, max_wait=60)

            return {
                "pipeline_id": pipeline.pipeline_id,
                "collection": pipeline.collection_name,
                "status": "timeout",
                "error": "Ingestion timeout",
                "ingestion_time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Ingestion failed: {e}")

            # Clean up worker only (keep API and infrastructure running)
            subprocess.run(["docker", "compose", "stop", "worker"], capture_output=True, timeout=30)
            self.wait_for_gpu_memory_clear(threshold_percent=50.0, max_wait=60)

            return {
                "pipeline_id": pipeline.pipeline_id,
                "collection": pipeline.collection_name,
                "status": "failed",
                "error": str(e),
                "ingestion_time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

    def wait_for_api_ready(self, max_wait: int = API_HEALTH_TIMEOUT) -> bool:
        """Wait for RAG API to be ready"""
        print(f"\n   Waiting for API to be ready (max {max_wait}s)...")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(RAG_API_HEALTH, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        print(f"   ✓ API is ready (took {time.time() - start_time:.1f}s)")
                        return True
            except Exception:
                pass

            time.sleep(2)

        print(f"   ⚠️  API did not become ready within {max_wait}s")
        return False

    def measure_api_baseline_memory(self) -> None:
        """Measure GPU memory usage with API running (one-time at startup)"""
        print("\n   📊 Measuring API baseline GPU memory...")
        time.sleep(3)  # Let API settle
        result = self.check_gpu_memory_free()
        if result:
            free_mb, total_mb = result
            used_mb = total_mb - free_mb
            self.api_baseline_memory_mb = used_mb
            print(f"   ✓ API baseline: {used_mb} MiB ({(used_mb/total_mb*100):.1f}% of {total_mb} MiB)")
        else:
            # Fallback estimate
            self.api_baseline_memory_mb = 2500  # ~2.5GB as rough default
            print(f"   ⚠️  Could not measure baseline, using estimate: {self.api_baseline_memory_mb} MiB")

    def check_api_health(self) -> bool:
        """Quick health check of API without waiting"""
        try:
            response = requests.get(RAG_API_HEALTH, timeout=3)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except Exception:
            pass
        return False

    def restart_api_if_needed(self, pipeline: PipelineConfig) -> bool:
        """
        Smart API lifecycle management:
        - Restart API if embedding model, retrieval strategy, or reranking strategy changed
        - Start API if it is not running yet

        Returns:
            True if API is ready, False if startup failed
        """
        embedding_provider = pipeline.embedding["provider"]
        embedding_model = pipeline.embedding["model"]
        embedding_key = f"{embedding_provider}:{embedding_model}"

        needed_retrieval = pipeline.retrieval["strategy"]       # "dense" or "hybrid"
        needed_reranking = pipeline.retrieval["reranking"]      # "none" or "cross-encoder"

        # Decide whether we need to (re)start rag-api
        restart_needed = (
            self.current_api_embedding != embedding_key or
            self.current_api_retrieval != needed_retrieval or
            self.current_api_reranking != needed_reranking
        )

        if not restart_needed:
            # API config matches this pipeline, just check health
            print(f"\n✓ Reusing rag-api (embedding={embedding_key}, retrieval={needed_retrieval}, reranking={needed_reranking})")
            if self.check_api_health():
                return True

            print("   ⚠️  API health check failed, attempting recovery with same config...")

        # Either config changed or health failed → hard restart rag-api with new env
        print("\n⏳ (Re)starting rag-api with config:")
        print(f"   embedding  = {embedding_key}")
        print(f"   retrieval  = {needed_retrieval}")
        print(f"   reranking  = {needed_reranking}")

        # Stop & remove old rag-api container (if any)
        subprocess.run(
            ["docker", "compose", "rm", "-sf", "rag-api"],
            capture_output=True,
            text=True,
        )

        # Build environment for this run
        env = os.environ.copy()
        env["EMBEDDING_PROVIDER"] = embedding_provider
        env["EMBEDDING_MODEL"] = embedding_model
        env["RETRIEVAL_STRATEGY"] = needed_retrieval
        env["RERANKING_STRATEGY"] = needed_reranking

        # For OpenAI, also set OPENAI_EMBEDDING_MODEL
        if embedding_provider == "openai":
            env["OPENAI_EMBEDDING_MODEL"] = embedding_model

        # Start rag-api with these env vars
        subprocess.run(
            ["docker", "compose", "up", "-d", "rag-api"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

        if not self.wait_for_api_ready(max_wait=API_HEALTH_TIMEOUT):
            print("   ❌ rag-api did not become ready after restart")
            return False

        # Update current config trackers
        self.current_api_embedding = embedding_key
        self.current_api_retrieval = needed_retrieval
        self.current_api_reranking = needed_reranking

        # Measure baseline memory once (after first successful start)
        if self.api_baseline_memory_mb is None:
            self.measure_api_baseline_memory()

        print("   ✓ rag-api ready")
        return True

    def run_retrieval_test(self, pipeline: PipelineConfig) -> Dict[str, Any]:
        """
        Run retrieval tests for a pipeline configuration.

        Returns:
            Result dict with retrieval metrics
        """
        print(f"\n{'='*80}")
        print(f"RETRIEVAL TEST: {pipeline.pipeline_id}")
        print(f"{'='*80}")

        try:
            # Step 0: Verify/restart API if needed (smart lifecycle management)
            if not self.restart_api_if_needed(pipeline):
                return {
                    "pipeline_id": pipeline.pipeline_id,
                    "collection": pipeline.collection_name,
                    "status": "failed",
                    "error": "API startup/recovery failed",
                    "timestamp": datetime.now().isoformat(),
                }

            # Step 1: Run test queries
            print(f"\n1️⃣  Running {len(self.test_queries)} test queries...")
            query_results = []
            total_retrieval_time = 0

            for i, query_data in enumerate(self.test_queries):
                query = query_data["query"]
                print(f"\n   Query {i+1}/{len(self.test_queries)}: {query[:60]}...")

                try:
                    start_time = time.time()
                    response = requests.post(
                        RAG_API_QUERY,
                        json={
                            "query": query,
                            "top_k": 5,
                            "min_score": 0.3,
                            "collection": pipeline.collection_name  # Override collection per test
                        },
                        timeout=60,
                    )
                    retrieval_time = time.time() - start_time

                    if response.status_code == 200:
                        data = response.json()
                        sources = data.get("sources", [])
                        answer = data.get("answer", "")
                        contexts = [s.get("text", "") for s in sources]
                        avg_score = sum(s["score"] for s in sources) / len(sources) if sources else 0
                        retrieval_meta = data.get("retrieval", {})

                        query_result = {
                            "query_id": query_data["id"],
                            "query": query,
                            "query_type": query_data.get("type", "unknown"),
                            "status": "success",
                            "results_count": len(sources),
                            "avg_score": avg_score,
                            "retrieval_time_ms": retrieval_time * 1000,
                            "scores": [s["score"] for s in sources],
                            # Answer and context data for later evaluation
                            "answer": answer,
                            "contexts": contexts,
                            "retrieval": retrieval_meta,
                        }

                        print(f"      ✓ Found {len(sources)} results (avg score: {avg_score:.3f})")
                        print(f"      ✓ Answer generated ({len(answer)} chars)")
                        total_retrieval_time += retrieval_time

                    else:
                        query_result = {
                            "query_id": query_data["id"],
                            "query": query,
                            "status": "failed",
                            "error": f"HTTP {response.status_code}",
                        }
                        print(f"      ✗ Failed: HTTP {response.status_code}")

                except Exception as e:
                    query_result = {
                        "query_id": query_data["id"],
                        "query": query,
                        "status": "error",
                        "error": str(e),
                    }
                    print(f"      ✗ Error: {e}")

                query_results.append(query_result)

            # Calculate aggregate metrics
            successful_queries = [q for q in query_results if q.get("status") == "success"]

            first_meta = None
            for q in successful_queries:
                if "retrieval" in q:
                    first_meta = q["retrieval"]
                    break

            result = {
                "pipeline_id": pipeline.pipeline_id,
                "collection": pipeline.collection_name,
                "status": "success",
                "queries_tested": len(self.test_queries),
                "queries_successful": len(successful_queries),
                "avg_retrieval_time_ms": (total_retrieval_time * 1000 / len(successful_queries)) if successful_queries else 0,
                "avg_score": sum(q["avg_score"] for q in successful_queries) / len(successful_queries) if successful_queries else 0,
                "queries": query_results,
                "retrieval": first_meta,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"\n✅ Retrieval test completed")
            print(f"   Successful: {len(successful_queries)}/{len(self.test_queries)}")
            print(f"   Avg score: {result['avg_score']:.3f}")
            print(f"   Avg retrieval time: {result['avg_retrieval_time_ms']:.0f}ms")

            return result

        except Exception as e:
            print(f"\n❌ Retrieval test failed: {e}")
            return {
                "pipeline_id": pipeline.pipeline_id,
                "collection": pipeline.collection_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def save_results(self) -> None:
        """Save all results to JSON file"""
        output = {
            "test_run": datetime.now().isoformat(),
            "total_pipelines": len(self.results),
            "results": self.results,
        }

        with open(self.results_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n💾 Results saved to {self.results_file}")

    def run_all_tests(self, pipelines: List[PipelineConfig], start_from: int = 0) -> None:
        """Run tests for all pipelines"""
        print("="*80)
        print("PIPELINE TESTING AUTOMATION")
        print("="*80)
        print(f"\nTotal pipelines: {len(pipelines)}")
        print(f"Starting from: {start_from + 1}")
        print(f"Estimated time: {(len(pipelines) - start_from) * 15 / 60:.1f} hours")
        print(f"Results file: {self.results_file}")

        # ===== START INFRASTRUCTURE ONCE =====
        print("\n" + "="*80)
        print("⏳ Starting RAG infrastructure (Qdrant, Postgres, Ollama, Minio)")
        print("="*80)

        print("\n1️⃣  Starting infrastructure (qdrant, postgres, ollama, minio)...")
        subprocess.run(
            ["docker", "compose", "up", "-d", "qdrant", "postgres", "ollama", "minio"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        time.sleep(5)  # Let services initialize

        print("\n✅ Infrastructure ready")
        print("   - rag-api will be (re)started automatically per pipeline as needed")
        print("   - Collections switched via query parameter for retrieval")
        print("   - Worker cleaned up after each test (GPU memory freed)\n")

        # =============================================

        for i, pipeline in enumerate(pipelines[start_from:], start=start_from):
            print(f"\n\n{'#'*80}")
            print(f"PIPELINE {i+1}/{len(pipelines)}: {pipeline.pipeline_id}")
            print(f"{'#'*80}")

            # Phase 1: Ingestion
            ingestion_result = self.run_ingestion(pipeline)
            self.results.append({"phase": "ingestion", **ingestion_result})

            # Save intermediate results
            self.save_results()

            # Phase 2: Retrieval (only if ingestion succeeded)
            if ingestion_result["status"] == "success":
                retrieval_result = self.run_retrieval_test(pipeline)
                self.results.append({"phase": "retrieval", **retrieval_result})

                # Save intermediate results
                self.save_results()
            else:
                print(f"\n⏭️  Skipping retrieval test (ingestion failed)")

        # Final summary
        self.print_summary()

    def print_summary(self) -> None:
        """Print test summary"""
        print(f"\n\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")

        ingestion_results = [r for r in self.results if r.get("phase") == "ingestion"]
        retrieval_results = [r for r in self.results if r.get("phase") == "retrieval"]

        ingestion_success = sum(1 for r in ingestion_results if r.get("status") == "success")
        retrieval_success = sum(1 for r in retrieval_results if r.get("status") == "success")

        print(f"\nIngestion:")
        print(f"  Total: {len(ingestion_results)}")
        print(f"  Successful: {ingestion_success}")
        print(f"  Failed: {len(ingestion_results) - ingestion_success}")

        print(f"\nRetrieval:")
        print(f"  Total: {len(retrieval_results)}")
        print(f"  Successful: {retrieval_success}")
        print(f"  Failed: {len(retrieval_results) - retrieval_success}")

        print(f"\nResults saved to: {self.results_file}")


def main():
    """Main test orchestration"""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline testing automation")
    parser.add_argument("--dry-run", action="store_true", help="List pipelines without running")
    parser.add_argument("--start-from", type=int, default=0, help="Start from pipeline N (0-indexed)")
    parser.add_argument("--limit", type=int, help="Only test N pipelines")
    parser.add_argument("--results-file", default="pipeline_test_results.json", help="Results output file")

    args = parser.parse_args()

    tester = PipelineTester(results_file=args.results_file)
    pipelines = tester.generate_all_pipelines()

    if args.dry_run:
        print(f"Total pipeline combinations: {len(pipelines)}\n")
        for i, pipeline in enumerate(pipelines):
            print(f"{i+1:3d}. {pipeline.pipeline_id:25s} → {pipeline.collection_name}")
        return

    # Apply limit if specified
    if args.limit:
        pipelines = pipelines[args.start_from:args.start_from + args.limit]
    else:
        pipelines = pipelines[args.start_from:]

    # Confirm before running
    response = input(f"\n🚀 Run tests for {len(pipelines)} pipelines? [y/N]: ")
    if response.lower() != "y":
        print("Aborted.")
        return

    # Run all tests
    tester.run_all_tests(pipelines, start_from=args.start_from)


if __name__ == "__main__":
    main()
