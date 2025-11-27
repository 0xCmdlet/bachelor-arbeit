#!/usr/bin/env python3
"""
Pipeline Results Analyzer

Analyzes pipeline test results and generates comparison reports.
Identifies best-performing pipelines across different metrics.
"""

import json
import sys
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime


class ResultsAnalyzer:
    """Analyzes pipeline test results"""

    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = []
        self.ingestion_results = []
        self.retrieval_results = []

        self._load_results()

    def _load_results(self) -> None:
        """Load results from JSON file"""
        try:
            with open(self.results_file, "r") as f:
                data = json.load(f)
                self.results = data.get("results", [])

            # Separate by phase
            self.ingestion_results = [r for r in self.results if r.get("phase") == "ingestion"]
            self.retrieval_results = [r for r in self.results if r.get("phase") == "retrieval"]

            print(f"✓ Loaded {len(self.results)} results from {self.results_file}")
            print(f"  Ingestion results: {len(self.ingestion_results)}")
            print(f"  Retrieval results: {len(self.retrieval_results)}")

        except Exception as e:
            print(f"Error loading results: {e}", file=sys.stderr)
            sys.exit(1)

    def generate_summary(self) -> str:
        """Generate markdown summary report"""
        lines = []
        lines.append("# Pipeline Testing Results Summary")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Results file**: {self.results_file}")
        lines.append("")

        # Overall statistics
        lines.append("## Overall Statistics")
        lines.append("")

        ingestion_success = [r for r in self.ingestion_results if r.get("status") == "success"]
        retrieval_success = [r for r in self.retrieval_results if r.get("status") == "success"]

        lines.append(f"- **Total pipelines tested**: {len(self.ingestion_results)}")
        lines.append(f"- **Successful ingestion**: {len(ingestion_success)}/{len(self.ingestion_results)} ({len(ingestion_success)/len(self.ingestion_results)*100:.1f}%)")
        lines.append(f"- **Successful retrieval**: {len(retrieval_success)}/{len(self.retrieval_results)} ({len(retrieval_success)/len(self.retrieval_results)*100:.1f}% of tested)")
        lines.append("")

        # Ingestion metrics
        if ingestion_success:
            avg_time = sum(r["ingestion_time_seconds"] for r in ingestion_success) / len(ingestion_success)
            min_time = min(r["ingestion_time_seconds"] for r in ingestion_success)
            max_time = max(r["ingestion_time_seconds"] for r in ingestion_success)

            lines.append("### Ingestion Metrics")
            lines.append("")
            lines.append(f"- **Average ingestion time**: {avg_time:.1f}s ({avg_time/60:.1f}m)")
            lines.append(f"- **Fastest ingestion**: {min_time:.1f}s")
            lines.append(f"- **Slowest ingestion**: {max_time:.1f}s")
            lines.append("")

        # Retrieval metrics
        if retrieval_success:
            avg_score = sum(r.get("avg_score", 0) for r in retrieval_success) / len(retrieval_success)
            avg_time = sum(r.get("avg_retrieval_time_ms", 0) for r in retrieval_success) / len(retrieval_success)

            lines.append("### Retrieval Metrics")
            lines.append("")
            lines.append(f"- **Average retrieval score**: {avg_score:.3f}")
            lines.append(f"- **Average retrieval time**: {avg_time:.0f}ms")
            lines.append("")

        # Answer quality metrics (NEW)
        if retrieval_success:
            # Collect all answers from queries
            all_answers = []
            for result in retrieval_success:
                queries = result.get("queries", [])
                for q in queries:
                    if "answer" in q:
                        all_answers.append(q["answer"])

            if all_answers:
                lines.append("### Answer Generation Metrics")
                lines.append("")

                non_empty = [a for a in all_answers if a and a.strip()]
                avg_length = sum(len(a) for a in non_empty) / len(non_empty) if non_empty else 0

                lines.append(f"- **Total answers generated**: {len(all_answers)}")
                lines.append(f"- **Non-empty answers**: {len(non_empty)}/{len(all_answers)} ({len(non_empty)/len(all_answers)*100:.1f}%)")
                lines.append(f"- **Average answer length**: {avg_length:.0f} characters")

                if non_empty:
                    min_length = min(len(a) for a in non_empty)
                    max_length = max(len(a) for a in non_empty)
                    lines.append(f"- **Answer length range**: {min_length}-{max_length} characters")

                lines.append("")

        # Top performers
        lines.append("## Top Performing Pipelines")
        lines.append("")

        if retrieval_success:
            # By average score
            lines.append("### By Average Retrieval Score (Top 10)")
            lines.append("")
            lines.append("| Rank | Pipeline ID | Collection | Avg Score | Retrieval Time |")
            lines.append("|------|-------------|------------|-----------|----------------|")

            top_by_score = sorted(retrieval_success, key=lambda x: x.get("avg_score", 0), reverse=True)[:10]
            for i, result in enumerate(top_by_score, 1):
                lines.append(
                    f"| {i} | `{result['pipeline_id']}` | `{result['collection'][:40]}` | "
                    f"{result.get('avg_score', 0):.3f} | {result.get('avg_retrieval_time_ms', 0):.0f}ms |"
                )

            lines.append("")

            # By retrieval time (fastest)
            lines.append("### By Retrieval Speed (Top 10)")
            lines.append("")
            lines.append("| Rank | Pipeline ID | Collection | Avg Score | Retrieval Time |")
            lines.append("|------|-------------|------------|-----------|----------------|")

            top_by_speed = sorted(retrieval_success, key=lambda x: x.get("avg_retrieval_time_ms", float('inf')))[:10]
            for i, result in enumerate(top_by_speed, 1):
                lines.append(
                    f"| {i} | `{result['pipeline_id']}` | `{result['collection'][:40]}` | "
                    f"{result.get('avg_score', 0):.3f} | {result.get('avg_retrieval_time_ms', 0):.0f}ms |"
                )

            lines.append("")

        # Component analysis
        lines.append("## Component Analysis")
        lines.append("")

        if retrieval_success:
            # Analyze by extractor
            extractor_scores = self._analyze_by_component(retrieval_success, "extractor")
            lines.append("### Extractors")
            lines.append("")
            lines.append("| Extractor | Avg Score | Avg Time (ms) | Count |")
            lines.append("|-----------|-----------|---------------|-------|")
            for extractor, metrics in sorted(extractor_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True):
                lines.append(
                    f"| {extractor} | {metrics['avg_score']:.3f} | {metrics['avg_time']:.0f} | {metrics['count']} |"
                )
            lines.append("")

            # Analyze by chunking
            chunking_scores = self._analyze_by_component(retrieval_success, "chunking")
            lines.append("### Chunking Strategies")
            lines.append("")
            lines.append("| Strategy | Avg Score | Avg Time (ms) | Count |")
            lines.append("|----------|-----------|---------------|-------|")
            for strategy, metrics in sorted(chunking_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True):
                lines.append(
                    f"| {strategy} | {metrics['avg_score']:.3f} | {metrics['avg_time']:.0f} | {metrics['count']} |"
                )
            lines.append("")

            # Analyze by embedding
            embedding_scores = self._analyze_by_component(retrieval_success, "embedding")
            lines.append("### Embedding Providers")
            lines.append("")
            lines.append("| Provider | Avg Score | Avg Time (ms) | Count |")
            lines.append("|----------|-----------|---------------|-------|")
            for provider, metrics in sorted(embedding_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True):
                lines.append(
                    f"| {provider} | {metrics['avg_score']:.3f} | {metrics['avg_time']:.0f} | {metrics['count']} |"
                )
            lines.append("")

            # Analyze by retrieval
            retrieval_scores = self._analyze_by_component(retrieval_success, "retrieval")
            lines.append("### Retrieval Strategies")
            lines.append("")
            lines.append("| Strategy | Avg Score | Avg Time (ms) | Count |")
            lines.append("|----------|-----------|---------------|-------|")
            for strategy, metrics in sorted(retrieval_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True):
                # Combine strategy + reranking
                lines.append(
                    f"| {strategy} | {metrics['avg_score']:.3f} | {metrics['avg_time']:.0f} | {metrics['count']} |"
                )
            lines.append("")

        # Failed pipelines
        ingestion_failed = [r for r in self.ingestion_results if r.get("status") != "success"]
        retrieval_failed = [r for r in self.retrieval_results if r.get("status") != "success"]

        if ingestion_failed or retrieval_failed:
            lines.append("## Failed Pipelines")
            lines.append("")

            if ingestion_failed:
                lines.append("### Ingestion Failures")
                lines.append("")
                lines.append("| Pipeline ID | Status | Error |")
                lines.append("|-------------|--------|-------|")
                for result in ingestion_failed:
                    error = result.get("error", "Unknown")[:50]
                    lines.append(f"| `{result['pipeline_id']}` | {result['status']} | {error} |")
                lines.append("")

            if retrieval_failed:
                lines.append("### Retrieval Failures")
                lines.append("")
                lines.append("| Pipeline ID | Status | Error |")
                lines.append("|-------------|--------|-------|")
                for result in retrieval_failed:
                    error = result.get("error", "Unknown")[:50]
                    lines.append(f"| `{result['pipeline_id']}` | {result['status']} | {error} |")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if retrieval_success:
            best = retrieval_success[0] if retrieval_success else None
            if best:
                best_by_score = max(retrieval_success, key=lambda x: x.get("avg_score", 0))
                best_by_speed = min(retrieval_success, key=lambda x: x.get("avg_retrieval_time_ms", float('inf')))

                lines.append(f"**Best overall quality** (highest avg score): `{best_by_score['pipeline_id']}`")
                lines.append(f"- Score: {best_by_score.get('avg_score', 0):.3f}")
                lines.append(f"- Speed: {best_by_score.get('avg_retrieval_time_ms', 0):.0f}ms")
                lines.append("")

                lines.append(f"**Best speed** (fastest retrieval): `{best_by_speed['pipeline_id']}`")
                lines.append(f"- Score: {best_by_speed.get('avg_score', 0):.3f}")
                lines.append(f"- Speed: {best_by_speed.get('avg_retrieval_time_ms', 0):.0f}ms")
                lines.append("")

        return "\n".join(lines)

    def _analyze_by_component(self, results: List[Dict[str, Any]], component: str) -> Dict[str, Dict[str, float]]:
        """
        Analyze results grouped by a component (extractor, chunking, etc.)

        Returns:
            Dict of {component_value: {avg_score, avg_time, count}}
        """
        groups = defaultdict(lambda: {"scores": [], "times": [], "count": 0})

        for result in results:
            config = result.get("config", {})
            component_value = config.get(component, "unknown")

            # For retrieval, combine strategy + reranking
            if component == "retrieval":
                strategy = config.get("retrieval", "unknown")
                reranking = config.get("reranking", "none")
                if reranking != "none":
                    component_value = f"{strategy}+{reranking}"
                else:
                    component_value = strategy

            groups[component_value]["scores"].append(result.get("avg_score", 0))
            groups[component_value]["times"].append(result.get("avg_retrieval_time_ms", 0))
            groups[component_value]["count"] += 1

        # Calculate averages
        summary = {}
        for key, data in groups.items():
            summary[key] = {
                "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0,
                "avg_time": sum(data["times"]) / len(data["times"]) if data["times"] else 0,
                "count": data["count"],
            }

        return summary

    def export_csv(self, output_file: str = "pipeline_results.csv") -> None:
        """Export results to CSV for further analysis"""
        import csv

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "pipeline_id",
                "extractor",
                "chunking",
                "embedding",
                "retrieval",
                "reranking",
                "ingestion_status",
                "ingestion_time_s",
                "retrieval_status",
                "avg_score",
                "avg_retrieval_time_ms",
                "queries_successful",
            ])

            # Match ingestion and retrieval results by pipeline_id
            ingestion_by_id = {r["pipeline_id"]: r for r in self.ingestion_results}
            retrieval_by_id = {r["pipeline_id"]: r for r in self.retrieval_results}

            all_pipeline_ids = set(ingestion_by_id.keys()) | set(retrieval_by_id.keys())

            for pipeline_id in sorted(all_pipeline_ids):
                ing = ingestion_by_id.get(pipeline_id, {})
                ret = retrieval_by_id.get(pipeline_id, {})

                config = ing.get("config") or ret.get("config", {})

                writer.writerow([
                    pipeline_id,
                    config.get("extractor", ""),
                    config.get("chunking", ""),
                    config.get("embedding", ""),
                    config.get("retrieval", ""),
                    config.get("reranking", ""),
                    ing.get("status", "not_run"),
                    ing.get("ingestion_time_seconds", 0),
                    ret.get("status", "not_run"),
                    ret.get("avg_score", 0),
                    ret.get("avg_retrieval_time_ms", 0),
                    ret.get("queries_successful", 0),
                ])

        print(f"✓ Exported results to {output_file}")


def main():
    """Main analyzer"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze pipeline test results")
    parser.add_argument("results_file", help="Path to pipeline_test_results.json")
    parser.add_argument("--output", "-o", default="pipeline_summary.md", help="Output markdown file")
    parser.add_argument("--csv", help="Also export to CSV")
    parser.add_argument("--print", action="store_true", help="Print to stdout instead of file")

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.results_file)

    # Generate summary
    summary = analyzer.generate_summary()

    # Output
    if args.print:
        print("\n" + summary)
    else:
        with open(args.output, "w") as f:
            f.write(summary)
        print(f"\n✓ Summary report saved to {args.output}")

    # Export CSV if requested
    if args.csv:
        analyzer.export_csv(args.csv)


if __name__ == "__main__":
    main()
