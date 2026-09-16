#!/usr/bin/env python
"""Download the Shanghai metro GTFS from Transitland (feed ``f-shanghai~metro``).

Transitland now requires a free API key (https://www.transit.land/ -> sign up ->
"API keys"). Set it via the ``TRANSITLAND_API_KEY`` environment variable or
``--api-key``, then run once:

    python scripts/shanghai/download_shanghai_gtfs.py

The feed is saved to ``data/shanghai/gtfs/raw/shanghai-gtfs.zip`` with its
SHA256 recorded alongside (matching the Singapore/Helsinki provenance layout).

The feed is metro-only (Shanghai Metro); no official open bus GTFS exists for
Shanghai, so bus supply is absent — report this honestly in the covariate audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]

FEED_ONESTOP_ID = "f-shanghai~metro"
DOWNLOAD_URL = (
    f"https://transit.land/api/v2/rest/feeds/{FEED_ONESTOP_ID}/download_latest_feed_version"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("TRANSITLAND_API_KEY", ""))
    ap.add_argument("--out", default=str(_WB / "data/shanghai/gtfs/raw/shanghai-gtfs.zip"))
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: set TRANSITLAND_API_KEY (free key from https://www.transit.land/) "
              "or pass --api-key.", file=sys.stderr)
        return 2

    try:
        import httpx
    except ImportError:
        import urllib.request

        req = urllib.request.Request(DOWNLOAD_URL, headers={"apikey": args.api_key})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
    else:
        r = httpx.get(DOWNLOAD_URL, headers={"apikey": args.api_key}, follow_redirects=True, timeout=600)
        r.raise_for_status()
        data = r.content

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    (out.parent / "checksum.sha256").write_text(sha + "\n", encoding="utf-8")
    meta = {
        "source": "Transitland",
        "feed_onestop_id": FEED_ONESTOP_ID,
        "download_url": DOWNLOAD_URL,
        "bytes": len(data),
        "sha256": sha,
    }
    (out.parent / "source_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved {out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
