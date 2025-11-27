#!/usr/bin/env python3

import asyncio
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Load .env from same directory
load_dotenv(Path(__file__).parent / '.env')

from ragas import SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


def parse_pipeline_results(results_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse flat results array and group by pipeline_id.

    Input structure: Flat array with alternating ingestion/retrieval phases
    Output: List of dicts with paired ingestion + retrieval data
    """
    results_array = results_data.get("results", [])

    # Group by pipeline_id
    pipelines = {}

    for entry in results_array:
        pipeline_id = entry.get("pipeline_id")
        phase = entry.get("phase")

        if pipeline_id not in pipelines:
            pipelines[pipeline_id] = {}

        pipelines[pipeline_id][phase] = entry

    # Convert to list and filter to only successful retrievals
    parsed = []
    for pipeline_id, phases in pipelines.items():
        retrieval = phases.get("retrieval")
        ingestion = phases.get("ingestion", {})

        # Only include if retrieval phase exists and succeeded
        if retrieval and retrieval.get("status") == "success":
            parsed.append({
                "pipeline_id": pipeline_id,
                "ingestion": ingestion,
                "retrieval": retrieval,
                "config": ingestion.get("config", {}),
            })

    return parsed


async def evaluate_pipeline_results(
    results_file: Path,
    ground_truth_file: Optional[Path] = None,
    pipeline_ids: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    output_dir: Path = None,
    phase: int = 2
):
    """
    Evaluate stored pipeline results with RAGAS metrics.

    Args:
        results_file: Path to pipeline_test_results.json
        ground_truth_file: Optional ground truth mapping for queries
        pipeline_ids: Specific pipeline IDs to evaluate (None = all)
        top_n: Evaluate only top N pipelines by avg_score (None = all)
        output_dir: Where to save results
    """

    print("=" * 80)
    print("PIPELINE RESULTS EVALUATION WITH RAGAS")
    print("=" * 80)
    print(f"Results file: {results_file}")
    print(f"Ground truth: {ground_truth_file or 'Not provided (Faithfulness only)'}")
    print(f"Output dir: {output_dir}")

    # Load pipeline results
    print("\n" + "=" * 80)
    print("Loading pipeline results...")
    print("=" * 80)

    with open(results_file, 'r') as f:
        pipeline_data = json.load(f)

    parsed_pipelines = parse_pipeline_results(pipeline_data)
    print(f"✓ Loaded {len(parsed_pipelines)} pipelines with successful retrieval")

    if not parsed_pipelines:
        print("✗ No successful pipeline results to evaluate!")
        return None

    # Load ground truth if provided
    ground_truth_map = {}
    if ground_truth_file and ground_truth_file.exists():
        print(f"\n✓ Loading ground truth from {ground_truth_file}")
        with open(ground_truth_file, 'r') as f:
            gt_data = json.load(f)
            # Expected format: {"q1": "reference answer", "q2": "...", ...}
            # or [{"query_id": "q1", "reference": "..."}, ...]
            if isinstance(gt_data, dict):
                ground_truth_map = gt_data
            elif isinstance(gt_data, list):
                ground_truth_map = {item["query_id"]: item["reference"] for item in gt_data}
        print(f"  ✓ Loaded ground truth for {len(ground_truth_map)} queries")

    # Filter pipelines based on criteria
    if pipeline_ids:
        parsed_pipelines = [p for p in parsed_pipelines if p["pipeline_id"] in pipeline_ids]
        print(f"✓ Filtered to {len(parsed_pipelines)} pipelines by ID")

    if top_n:
        # Sort by average retrieval score and take top N
        parsed_pipelines = sorted(
            parsed_pipelines,
            key=lambda p: p["retrieval"].get("avg_score", 0),
            reverse=True
        )[:top_n]
        print(f"✓ Evaluating top {top_n} pipelines by avg_score")

    print(f"\n→ Evaluating {len(parsed_pipelines)} pipelines")

    # Initialize metrics
    print("\n" + "=" * 80)
    print(f"Initializing RAGAS metrics (Phase {phase})...")
    print("=" * 80)

    metrics = {}

    if phase == 1:
        # Phase 1: Only semantic similarity (no LLM needed)
        if not ground_truth_map:
            print("✗ Error: Phase 1 requires ground truth for semantic_similarity")
            return None

        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        print(f"✓ Using embeddings: {EMBEDDING_MODEL}")

        metrics = {
            "semantic_similarity": SemanticSimilarity(embeddings=embeddings)
        }
        print(f"✓ Phase 1: Initialized semantic_similarity metric (no LLM calls)")

    elif phase == 2:
        # Phase 2: All metrics (current behavior)
        openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm = LangchainLLMWrapper(openai_llm)
        print(f"✓ Using LLM: gpt-4o-mini")

        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        print(f"✓ Using embeddings: {EMBEDDING_MODEL}")

        # Metrics: Faithfulness always available
        metrics = {
            "faithfulness": Faithfulness(llm=llm),
        }

        # Add reference-based metrics if ground truth available
        if ground_truth_map:
            metrics.update({
                "context_precision": LLMContextPrecisionWithReference(llm=llm),
                "factual_correctness": FactualCorrectness(llm=llm),
                "semantic_similarity": SemanticSimilarity(embeddings=embeddings)
            })
            print(f"✓ Phase 2: Initialized {len(metrics)} metrics (with ground truth)")
        else:
            print(f"⚠ Ground truth not provided - only Faithfulness metric available")
            print(f"  To enable all metrics, provide --ground-truth file with:")
            print(f"  {{'q1': 'reference answer', 'q2': '...'}}")
            print(f"✓ Phase 2: Initialized {len(metrics)} metric")

    # Evaluate each pipeline
    print("\n" + "=" * 80)
    print("Evaluating pipelines...")
    print("=" * 80)

    evaluation_results = []

    for pipeline in parsed_pipelines:
        pipeline_id = pipeline["pipeline_id"]
        config = pipeline["config"]
        retrieval = pipeline["retrieval"]
        queries = retrieval.get("queries", [])

        print(f"\n{'='*80}")
        print(f"Pipeline {pipeline_id}: {config['extractor']} + {config['chunking']} + {config['embedding']} + {config['retrieval']}")
        print(f"{'='*80}")

        if not queries:
            print("  ⚠ No queries found, skipping")
            continue

        # Evaluate each query
        pipeline_scores = {metric: [] for metric in metrics.keys()}
        query_details = []

        for i, query_data in enumerate(queries, 1):
            if query_data.get("status") != "success":
                continue

            query_id = query_data.get("query_id")
            query = query_data.get("query")
            answer = query_data.get("answer", "")
            contexts = query_data.get("contexts", [])

            print(f"  [{i}/{len(queries)}] {query_id}: {query[:60]}...")

            if not answer:
                print(f"    ⚠ No answer stored, skipping")
                continue

            if not contexts:
                print(f"    ⚠ No contexts stored, skipping")
                continue

            # Get ground truth if available
            reference = ground_truth_map.get(query_id, "")

            # Create RAGAS sample
            try:
                sample = SingleTurnSample(
                    user_input=query,
                    reference=reference,
                    retrieved_contexts=contexts,
                    response=answer
                )

                # Evaluate with metrics
                query_scores = {}
                for metric_name, metric in metrics.items():
                    try:
                        score = await metric.single_turn_ascore(sample)
                        query_scores[metric_name] = float(score)
                        pipeline_scores[metric_name].append(float(score))
                        print(f"      • {metric_name}: {score:.4f}")
                    except Exception as e:
                        print(f"      ✗ {metric_name} failed: {e}")
                        query_scores[metric_name] = None

                query_details.append({
                    "query_id": query_id,
                    "query": query,
                    "answer_length": len(answer),
                    "num_contexts": len(contexts),
                    "scores": query_scores
                })

            except Exception as e:
                print(f"    ✗ Evaluation failed: {e}")
                continue

        # Calculate pipeline averages
        pipeline_avg_scores = {}
        for metric_name, scores in pipeline_scores.items():
            if scores:
                pipeline_avg_scores[metric_name] = sum(scores) / len(scores)
            else:
                pipeline_avg_scores[metric_name] = None

        print(f"\n  Pipeline {pipeline_id} Average Scores:")
        for metric_name, avg_score in pipeline_avg_scores.items():
            if avg_score is not None:
                print(f"    {metric_name}: {avg_score:.4f}")
            else:
                print(f"    {metric_name}: N/A")

        evaluation_results.append({
            "pipeline_id": pipeline_id,
            "config": config,
            "retrieval_metrics": {
                "avg_retrieval_score": retrieval.get("avg_score", 0),
                "avg_retrieval_time_ms": retrieval.get("avg_retrieval_time_ms", 0),
                "queries_count": len([q for q in queries if q.get("status") == "success"])
            },
            "ragas_scores": pipeline_avg_scores,
            "query_details": query_details
        })

    # Generate comparison report
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    if not evaluation_results:
        print("✗ No pipelines were successfully evaluated!")
        return None

    # Sort by composite score (phase-dependent calculation)
    def composite_score(result):
        if phase == 1:
            # Phase 1: (semantic_similarity + avg_retrieval_score) / 2
            sem_sim = result["ragas_scores"].get("semantic_similarity")
            ret_score = result["retrieval_metrics"].get("avg_retrieval_score", 0)
            if sem_sim is not None:
                return (sem_sim + ret_score) / 2
            return 0
        else:
            # Phase 2: Average of all RAGAS metrics
            scores = [s for s in result["ragas_scores"].values() if s is not None]
            return sum(scores) / len(scores) if scores else 0

    evaluation_results.sort(key=composite_score, reverse=True)

    print(f"\nTop Pipelines by RAGAS Composite Score:")
    print(f"{'='*80}")
    for i, result in enumerate(evaluation_results[:10], 1):
        config = result["config"]
        comp_score = composite_score(result)
        print(f"\n{i}. Pipeline {result['pipeline_id']} - Composite: {comp_score:.4f}")
        print(f"   Config: {config['extractor']} + {config['chunking']} + {config['embedding']} + {config['retrieval']}")
        print(f"   RAGAS Scores:")
        for metric, score in result["ragas_scores"].items():
            if score is not None:
                print(f"     • {metric}: {score:.4f}")
        print(f"   Retrieval: avg_score={result['retrieval_metrics']['avg_retrieval_score']:.3f}, "
              f"avg_time={result['retrieval_metrics']['avg_retrieval_time_ms']:.0f}ms")

    # Phase 1: Save top 10 pipeline IDs for phase 2
    if phase == 1:
        top_10_ids = [r["pipeline_id"] for r in evaluation_results[:10]]
        phase1_output = output_dir / "phase1_top_pipelines.json"
        with open(phase1_output, 'w') as f:
            json.dump({
                "top_10_pipelines": top_10_ids,
                "timestamp": datetime.now().isoformat(),
                "scores": [
                    {
                        "pipeline_id": r["pipeline_id"],
                        "combined_score": composite_score(r),
                        "semantic_similarity": r["ragas_scores"].get("semantic_similarity"),
                        "avg_retrieval_score": r["retrieval_metrics"]["avg_retrieval_score"]
                    }
                    for r in evaluation_results[:10]
                ]
            }, f, indent=2)
        print(f"\n{'='*80}")
        print(f"✓ Phase 1 complete! Top 10 pipeline IDs saved to: {phase1_output}")
        print(f"{'='*80}")
        print(f"\nNext step: Run phase 2 with:")
        print(f"  python evaluate_pipeline_results.py --phase 2 --phase1-results {phase1_output} \\")
        print(f"    --results-file {results_file} --ground-truth {ground_truth_file or '<path>'}")

    # Save results
    save_evaluation_results(
        evaluation_results=evaluation_results,
        metrics_used=list(metrics.keys()),
        output_dir=output_dir,
        results_file=results_file,
        ground_truth_provided=bool(ground_truth_map),
        phase=phase
    )

    return evaluation_results


def save_evaluation_results(
    evaluation_results: List[Dict[str, Any]],
    metrics_used: List[str],
    output_dir: Path,
    results_file: Path,
    ground_truth_provided: bool,
    phase: int = 2
):
    """Save evaluation results to multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ragas_evaluation_phase{phase}_{timestamp}"

    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    # Save detailed JSON
    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source_file": str(results_file),
                "phase": phase,
                "metrics_used": metrics_used,
                "ground_truth_provided": ground_truth_provided,
                "num_pipelines_evaluated": len(evaluation_results)
            },
            "results": evaluation_results
        }, f, indent=2)
    print(f"✓ JSON: {json_path}")

    # Save comparison CSV
    csv_path = output_dir / f"{base_filename}_comparison.csv"
    csv_data = []
    for result in evaluation_results:
        row = {
            "pipeline_id": result["pipeline_id"],
            "extractor": result["config"]["extractor"],
            "chunking": result["config"]["chunking"],
            "embedding": result["config"]["embedding"],
            "retrieval": result["config"]["retrieval"],
            "avg_retrieval_score": result["retrieval_metrics"]["avg_retrieval_score"],
            "avg_retrieval_time_ms": result["retrieval_metrics"]["avg_retrieval_time_ms"],
        }
        # Add RAGAS scores
        for metric in metrics_used:
            row[f"ragas_{metric}"] = result["ragas_scores"].get(metric)

        # Add composite score
        scores = [s for s in result["ragas_scores"].values() if s is not None]
        row["ragas_composite"] = sum(scores) / len(scores) if scores else None

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV: {csv_path}")

    # Save summary report
    summary_path = output_dir / f"{base_filename}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RAGAS EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Source: {results_file}\n")
        f.write(f"Pipelines Evaluated: {len(evaluation_results)}\n")
        f.write(f"Metrics Used: {', '.join(metrics_used)}\n")
        f.write(f"Ground Truth: {'Yes' if ground_truth_provided else 'No'}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP 10 PIPELINES BY COMPOSITE SCORE\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(evaluation_results[:10], 1):
            config = result["config"]
            scores = [s for s in result["ragas_scores"].values() if s is not None]
            comp_score = sum(scores) / len(scores) if scores else 0

            f.write(f"{i}. Pipeline {result['pipeline_id']} - Composite: {comp_score:.4f}\n")
            f.write(f"   {config['extractor']} + {config['chunking']} + {config['embedding']} + {config['retrieval']}\n")
            f.write(f"   RAGAS Scores:\n")
            for metric, score in result["ragas_scores"].items():
                if score is not None:
                    f.write(f"     • {metric}: {score:.4f}\n")
            f.write(f"   Retrieval: score={result['retrieval_metrics']['avg_retrieval_score']:.3f}, ")
            f.write(f"time={result['retrieval_metrics']['avg_retrieval_time_ms']:.0f}ms\n\n")

    print(f"✓ Summary: {summary_path}")
    print(f"\n✓ All results saved to: {output_dir}")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline test results with RAGAS metrics"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("../results/pipeline_test_results.json"),
        help="Path to pipeline_test_results.json (default: ../results/pipeline_test_results.json)"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional ground truth file (JSON: {'q1': 'answer', ...})"
    )
    parser.add_argument(
        "--pipeline-ids",
        type=str,
        help="Comma-separated pipeline IDs to evaluate (e.g., '1,5,10')"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Evaluate only top N pipelines by retrieval score"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="Evaluation phase: 1=semantic+retrieval only (fast), 2=all metrics (default: 2)"
    )
    parser.add_argument(
        "--phase1-results",
        type=Path,
        help="Path to phase 1 results JSON (for phase 2, auto-loads top 10 pipeline IDs)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: ./results)"
    )

    args = parser.parse_args()

    # Parse pipeline IDs
    pipeline_ids = None
    if args.pipeline_ids:
        pipeline_ids = [x.strip() for x in args.pipeline_ids.split(",")]

    # Load phase 1 results if provided
    if args.phase1_results and args.phase1_results.exists():
        print(f"✓ Loading phase 1 results from: {args.phase1_results}")
        with open(args.phase1_results, 'r') as f:
            phase1_data = json.load(f)
            pipeline_ids = phase1_data.get("top_10_pipelines", [])
            print(f"  Loaded {len(pipeline_ids)} pipeline IDs: {', '.join(pipeline_ids)}")

    # Check for OpenAI API key (only required for phase 2)
    if args.phase == 2 and not os.getenv("OPENAI_API_KEY"):
        print("=" * 80)
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("=" * 80)
        print("\nPhase 2 metrics require an LLM for evaluation.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nNote: Phase 1 (semantic similarity only) does not require OpenAI API key")
        exit(1)

    # Check results file exists
    if not args.results_file.exists():
        print(f"✗ Error: Results file not found: {args.results_file}")
        print(f"  Please run test-pipelines.py first to generate results")
        exit(1)

    # Run evaluation
    try:
        results = await evaluate_pipeline_results(
            results_file=args.results_file,
            ground_truth_file=args.ground_truth,
            pipeline_ids=pipeline_ids,
            top_n=args.top_n,
            output_dir=args.output_dir,
            phase=args.phase
        )

        if results:
            print("\n" + "=" * 80)
            print("✓ Evaluation completed successfully!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠ Evaluation completed with no results")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
