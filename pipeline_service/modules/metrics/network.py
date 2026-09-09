from __future__ import annotations

import os
import time
import subprocess
import json
import urllib.request
from dataclasses import dataclass

MIN_DOWNLOAD_MBPS = float(os.environ.get("BENCHMARK_MIN_DOWNLOAD_MBPS", "100.0"))
MIN_UPLOAD_MBPS = float(os.environ.get("BENCHMARK_MIN_UPLOAD_MBPS", "0.0"))
import subprocess
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkBenchmarkResult:
    server_name: str
    server_country: str
    ping_ms: float
    download_mbps: float
    upload_mbps: float
    download_passed: bool
    upload_passed: bool
    passed: bool
    crashed: bool = False


ATTEMPTS = int(os.environ.get("BENCHMARK_NETWORK_ATTEMPTS", "3"))


def run_benchmark(
    min_download_mbps: float = MIN_DOWNLOAD_MBPS,
    min_upload_mbps: float = MIN_UPLOAD_MBPS,
) -> NetworkBenchmarkResult:
    best: NetworkBenchmarkResult | None = None
    for attempt in range(1, max(1, ATTEMPTS) + 1):
        r = _run_once(min_download_mbps, min_upload_mbps)
        if best is None or r.download_mbps > best.download_mbps:
            best = r
        if r.passed:
            if attempt > 1:
                print(f"[preflight] network passed on attempt {attempt} "
                      f"({r.download_mbps} Mbps down)")
            return r
        print(f"[preflight] network attempt {attempt}/{ATTEMPTS}: "
              f"{r.download_mbps} Mbps down (min {min_download_mbps})"
              + (" [crashed]" if r.crashed else ""))
    return best  # type: ignore[return-value]


def _run_once(
    min_download_mbps: float = MIN_DOWNLOAD_MBPS,
    min_upload_mbps: float = MIN_UPLOAD_MBPS,
) -> NetworkBenchmarkResult:
    try:
        result = subprocess.run(
            ['speedtest-cli', '--json', '--secure'],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(f"speedtest-cli failed: {result.stderr}")
            return NetworkBenchmarkResult(
                server_name="unknown",
                server_country="unknown",
                ping_ms=0.0,
                download_mbps=0.0,
                upload_mbps=0.0,
                download_passed=False,
                upload_passed=False,
                passed=False,
                crashed=True,
            )

        data = json.loads(result.stdout)
        download_mbps = round(data['download'] / 1_000_000, 1)
        upload_mbps = round(data['upload'] / 1_000_000, 1)
        ping_ms = round(data['ping'], 2)

        download_passed = download_mbps >= min_download_mbps
        upload_passed = upload_mbps >= min_upload_mbps

        return NetworkBenchmarkResult(
            server_name=data['server']['sponsor'],
            server_country=data['server']['country'],
            ping_ms=ping_ms,
            download_mbps=download_mbps,
            upload_mbps=upload_mbps,
            download_passed=download_passed,
            upload_passed=upload_passed,
            passed=download_passed and upload_passed,
        )

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.warning(f"speedtest benchmark failed: {e}")
        return NetworkBenchmarkResult(
            server_name="unknown",
            server_country="unknown",
            ping_ms=0.0,
            download_mbps=0.0,
            upload_mbps=0.0,
            download_passed=False,
            upload_passed=False,
            passed=False,
            crashed=True,
        )