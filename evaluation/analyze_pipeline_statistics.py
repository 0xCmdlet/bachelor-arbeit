#!/usr/bin/env python3
"""
Analyze full pipeline test results from JSON file.
"""
import json
import sys
from collections import defaultdict
from typing import Dict, List, Any


def load_results(filepath: str) -> Dict[str, Any]:
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_pipeline_results(results: Dict[str, Any]) -> None:
    """Analyze and print summary of pipeline test results."""

    print(f"\n{'='*80}")
    print(f"PIPELINE TEST RESULTS ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Test Run: {results['test_run']}")
    print(f"Total Pipelines: {results['total_pipelines']}")
    print()

    # Separate ingestion and retrieval results
    ingestion_results = []
    retrieval_results = []

    for result in results['results']:
        if result['phase'] == 'ingestion':
            ingestion_results.append(result)
        elif result['phase'] == 'retrieval':
            retrieval_results.append(result)

    print(f"Total Results: {len(results['results'])}")
    print(f"  - Ingestion: {len(ingestion_results)}")
    print(f"  - Retrieval: {len(retrieval_results)}")
    print()

    # Analyze ingestion phase
    print(f"\n{'='*80}")
    print("INGESTION PHASE ANALYSIS")
    print(f"{'='*80}\n")

    ingestion_success = sum(1 for r in ingestion_results if r['status'] == 'success')
    ingestion_failed = len(ingestion_results) - ingestion_success

    print(f"Success: {ingestion_success}/{len(ingestion_results)} ({ingestion_success/len(ingestion_results)*100:.1f}%)")
    print(f"Failed: {ingestion_failed}/{len(ingestion_results)} ({ingestion_failed/len(ingestion_results)*100:.1f}%)")

    if ingestion_success > 0:
        successful_ingestions = [r for r in ingestion_results if r['status'] == 'success']

        avg_time = sum(r['ingestion_time_seconds'] for r in successful_ingestions) / len(successful_ingestions)
        min_time = min(r['ingestion_time_seconds'] for r in successful_ingestions)
        max_time = max(r['ingestion_time_seconds'] for r in successful_ingestions)

        avg_chunks = sum(r.get('processed_count', 0) for r in successful_ingestions) / len(successful_ingestions)
        min_chunks = min(r.get('processed_count', 0) for r in successful_ingestions)
        max_chunks = max(r.get('processed_count', 0) for r in successful_ingestions)

        print(f"\nIngestion Time (seconds):")
        print(f"  Average: {avg_time:.1f}s")
        print(f"  Min: {min_time:.1f}s")
        print(f"  Max: {max_time:.1f}s")

        print(f"\nChunks Processed:")
        print(f"  Average: {avg_chunks:.0f}")
        print(f"  Min: {min_chunks}")
        print(f"  Max: {max_chunks}")

        # Find slowest and fastest
        slowest = max(successful_ingestions, key=lambda r: r['ingestion_time_seconds'])
        fastest = min(successful_ingestions, key=lambda r: r['ingestion_time_seconds'])

        print(f"\nSlowest Pipeline: {slowest['pipeline_id']} ({slowest['ingestion_time_seconds']:.1f}s)")
        print(f"  Config: {slowest['config']['extractor']}, {slowest['config']['chunking']}, {slowest['config']['embedding']}")

        print(f"\nFastest Pipeline: {fastest['pipeline_id']} ({fastest['ingestion_time_seconds']:.1f}s)")
        print(f"  Config: {fastest['config']['extractor']}, {fastest['config']['chunking']}, {fastest['config']['embedding']}")

    # Analyze retrieval phase
    print(f"\n{'='*80}")
    print("RETRIEVAL PHASE ANALYSIS")
    print(f"{'='*80}\n")

    retrieval_success = sum(1 for r in retrieval_results if r['status'] == 'success')
    retrieval_failed = len(retrieval_results) - retrieval_success

    print(f"Success: {retrieval_success}/{len(retrieval_results)} ({retrieval_success/len(retrieval_results)*100:.1f}%)")
    print(f"Failed: {retrieval_failed}/{len(retrieval_results)} ({retrieval_failed/len(retrieval_results)*100:.1f}%)")

    if retrieval_success > 0:
        successful_retrievals = [r for r in retrieval_results if r['status'] == 'success']

        avg_score = sum(r['avg_score'] for r in successful_retrievals) / len(successful_retrievals)
        min_score = min(r['avg_score'] for r in successful_retrievals)
        max_score = max(r['avg_score'] for r in successful_retrievals)

        avg_time = sum(r['avg_retrieval_time_ms'] for r in successful_retrievals) / len(successful_retrievals)
        min_time = min(r['avg_retrieval_time_ms'] for r in successful_retrievals)
        max_time = max(r['avg_retrieval_time_ms'] for r in successful_retrievals)

        print(f"\nAverage Retrieval Score:")
        print(f"  Overall Average: {avg_score:.4f}")
        print(f"  Min: {min_score:.4f}")
        print(f"  Max: {max_score:.4f}")

        print(f"\nAverage Retrieval Time (ms):")
        print(f"  Overall Average: {avg_time:.0f}ms")
        print(f"  Min: {min_time:.0f}ms")
        print(f"  Max: {max_time:.0f}ms")

        # Find best and worst performers
        best_score = max(successful_retrievals, key=lambda r: r['avg_score'])
        worst_score = min(successful_retrievals, key=lambda r: r['avg_score'])

        print(f"\nBest Scoring Pipeline: {best_score['pipeline_id']} (score: {best_score['avg_score']:.4f})")
        print(f"  Config: {best_score.get('config', 'N/A')}")

        print(f"\nWorst Scoring Pipeline: {worst_score['pipeline_id']} (score: {worst_score['avg_score']:.4f})")
        print(f"  Config: {worst_score.get('config', 'N/A')}")

        # Analyze by dimensions
        print(f"\n{'='*80}")
        print("PERFORMANCE BY DIMENSIONS")
        print(f"{'='*80}\n")

        # Group by extractor, chunking, embedding, retrieval
        by_extractor = defaultdict(list)
        by_chunking = defaultdict(list)
        by_embedding = defaultdict(list)
        by_retrieval = defaultdict(list)

        for r in successful_retrievals:
            # Get config from ingestion results
            pipeline_id = r['pipeline_id']
            ingestion = next((ing for ing in ingestion_results if ing['pipeline_id'] == pipeline_id), None)

            if ingestion and 'config' in ingestion:
                config = ingestion['config']
                by_extractor[config['extractor']].append(r)
                by_chunking[config['chunking']].append(r)
                by_embedding[config['embedding']].append(r)
                by_retrieval[config['retrieval']].append(r)

        print("\nBy Extractor:")
        for extractor, results in sorted(by_extractor.items()):
            avg = sum(r['avg_score'] for r in results) / len(results)
            print(f"  {extractor:15s}: {avg:.4f} (n={len(results)})")

        print("\nBy Chunking:")
        for chunking, results in sorted(by_chunking.items()):
            avg = sum(r['avg_score'] for r in results) / len(results)
            print(f"  {chunking:15s}: {avg:.4f} (n={len(results)})")

        print("\nBy Embedding:")
        for embedding, results in sorted(by_embedding.items()):
            avg = sum(r['avg_score'] for r in results) / len(results)
            print(f"  {embedding:30s}: {avg:.4f} (n={len(results)})")

        print("\nBy Retrieval Strategy:")
        for retrieval, results in sorted(by_retrieval.items()):
            avg = sum(r['avg_score'] for r in results) / len(results)
            print(f"  {retrieval:15s}: {avg:.4f} (n={len(results)})")

    # Check if any failures
    if ingestion_failed > 0 or retrieval_failed > 0:
        print(f"\n{'='*80}")
        print("FAILURES")
        print(f"{'='*80}\n")

        if ingestion_failed > 0:
            print("Failed Ingestions:")
            for r in ingestion_results:
                if r['status'] != 'success':
                    print(f"  - {r['pipeline_id']}: {r.get('error', 'Unknown error')}")

        if retrieval_failed > 0:
            print("\nFailed Retrievals:")
            for r in retrieval_results:
                if r['status'] != 'success':
                    print(f"  - {r['pipeline_id']}: {r.get('error', 'Unknown error')}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_json_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    results = load_results(filepath)
    analyze_pipeline_results(results)
