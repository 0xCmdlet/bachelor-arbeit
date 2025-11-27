#!/usr/bin/env python3
"""
Docker Log Monitor

Monitors Docker container logs for specific patterns and extracts metrics.
Used to detect when ingestion worker completes processing.
"""

import subprocess
import json
import time
import sys
from typing import Dict, Any, Optional
from datetime import datetime


class LogMonitor:
    """Monitors Docker container logs for completion patterns"""

    def __init__(self, container_name: str = "worker", timeout: int = 1200):
        """
        Initialize log monitor.

        Args:
            container_name: Name of Docker container to monitor
            timeout: Maximum time to wait in seconds (default: 20 minutes)
        """
        self.container_name = container_name
        self.timeout = timeout

    def wait_for_completion(self, completion_pattern: str = "worker_complete") -> Dict[str, Any]:
        """
        Wait for completion message in container logs.

        Args:
            completion_pattern: String pattern to look for in logs

        Returns:
            Dict with status, metrics, and timing info
        """
        start_time = time.time()
        result = {
            "status": "unknown",
            "elapsed_seconds": 0,
            "processed_count": 0,
            "error_count": 0,
            "completion_message": None,
        }

        print(f"Monitoring logs for container: {self.container_name}")
        print(f"Looking for pattern: {completion_pattern}")
        print(f"Timeout: {self.timeout}s ({self.timeout/60:.1f} minutes)")

        try:
            # Start following logs
            process = subprocess.Popen(
                ["docker", "compose", "logs", "-f", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            last_output_time = time.time()
            lines_checked = 0

            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout:
                    print(f"\n⏱️  Timeout reached ({self.timeout}s)")
                    result["status"] = "timeout"
                    result["elapsed_seconds"] = elapsed
                    process.kill()
                    break

                # Read line
                line = process.stdout.readline()

                if not line:
                    # No more output, check if process ended
                    if process.poll() is not None:
                        # Process ended
                        break
                    # Still running, just no output yet
                    time.sleep(0.1)
                    continue

                lines_checked += 1
                last_output_time = time.time()

                # Filter out verbose/noisy logs
                if any(skip in line for skip in [
                    "len(pages)=",
                    "len(valid_pages)=",
                    "len(valid_page_images)=",
                    "HTTP Request:",  # Qdrant API calls
                    "Embedding batch",  # Individual embedding batches
                    "tesserocr import failed",  # Debug warning (harmless)
                ]):
                    continue

                # Print line for visibility (truncate if too long)
                print_line = line.strip()
                if len(print_line) > 200:
                    print_line = print_line[:200] + "..."
                print(f"  {print_line}")

                # Check for completion pattern
                if completion_pattern in line:
                    print(f"\n✓ Found completion pattern: {completion_pattern}")

                    # Try to parse JSON metrics from log line
                    metrics = self._extract_metrics_from_line(line)
                    if metrics:
                        result["processed_count"] = metrics.get("processed_count", 0)
                        result["error_count"] = metrics.get("error_count", 0)
                        result["completion_message"] = metrics

                    result["status"] = "success"
                    result["elapsed_seconds"] = time.time() - start_time
                    process.kill()
                    break

                # Check for error patterns
                if "ERROR" in line or "error" in line.lower():
                    # Don't fail immediately, just note it
                    # Worker might continue processing other files
                    pass

            # Final status
            if result["status"] == "unknown":
                # Didn't find completion pattern and didn't timeout
                result["status"] = "failed"
                result["elapsed_seconds"] = time.time() - start_time

            print(f"\n📊 Monitoring complete:")
            print(f"   Status: {result['status']}")
            print(f"   Duration: {result['elapsed_seconds']:.1f}s")
            print(f"   Processed: {result['processed_count']}")
            print(f"   Errors: {result['error_count']}")
            print(f"   Lines checked: {lines_checked}")

            return result

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring interrupted by user")
            result["status"] = "interrupted"
            result["elapsed_seconds"] = time.time() - start_time
            return result

        except Exception as e:
            print(f"\n❌ Error monitoring logs: {e}", file=sys.stderr)
            result["status"] = "error"
            result["error_message"] = str(e)
            result["elapsed_seconds"] = time.time() - start_time
            return result

    def _extract_metrics_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON metrics from a log line.

        Worker logs are JSON formatted, so we try to parse them.
        """
        try:
            # Find JSON portion (might have timestamp prefix)
            json_start = line.find('{')
            if json_start == -1:
                return None

            json_str = line[json_start:]
            data = json.loads(json_str)
            return data

        except json.JSONDecodeError:
            # Not JSON or malformed
            return None
        except Exception:
            return None

    def check_container_running(self) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "-q", self.container_name],
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def get_container_status(self) -> str:
        """Get container status (running, exited, etc.)"""
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", self.container_name],
                capture_output=True,
                text=True,
                check=True,
            )
            # Parse output to get status
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # Second line has the status
                return lines[1] if len(lines) > 1 else "unknown"
            return "unknown"
        except Exception:
            return "error"


def main():
    """CLI interface for log monitor"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Docker container logs")
    parser.add_argument("--container", default="worker", help="Container name to monitor")
    parser.add_argument("--pattern", default="worker_complete", help="Completion pattern to look for")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds (default: 1200)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    monitor = LogMonitor(
        container_name=args.container,
        timeout=args.timeout
    )

    # Check if container exists
    if not monitor.check_container_running():
        print(f"⚠️  Container '{args.container}' is not running", file=sys.stderr)
        print(f"Status: {monitor.get_container_status()}")

    # Monitor for completion
    result = monitor.wait_for_completion(completion_pattern=args.pattern)

    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human readable output already printed during monitoring
        pass

    # Exit code based on status
    if result["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
