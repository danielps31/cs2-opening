"""Run everything for your account: fetch demos, parse, build the report.

    python -m src.pipeline            # Steam (QR login on first run) + CS2 replays folder -> report
    python -m src.pipeline --no-steam # only demos already on disk
    python -m src.pipeline --no-steam --gcpd "C:/Users/you/Downloads/Personal Game Data.html"
    python -m src.pipeline --no-steam --urls data/demo_urls.txt

The report opens on your steamid (from the Steam login, data/steam_account.json, or --me).
For a login page in the browser, use `python -m src.app`.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.config import ROOT
from src.data import collect, radar
from src.data.demo_parser import parse_demo_directory
from src.features import engineer
from src.report import build_report

DATA = ROOT / "data"
DEMOS = DATA / "raw" / "demos"
STEAM_TOOL = ROOT / "tools" / "steam_gc"


def ensure_node_tool() -> None:
    if not (STEAM_TOOL / "node_modules").exists():
        subprocess.run(["npm", "install", "--silent"], cwd=STEAM_TOOL, check=True, shell=sys.platform == "win32")


def download_url_file(urls_file: Path) -> None:
    urls = [u.strip() for u in urls_file.read_text().splitlines() if u.strip()] if urls_file.exists() else []
    print(f"Downloading {len(urls)} demos (already downloaded ones are skipped)...")
    collect.download_urls(urls, DEMOS)


def fetch_from_steam() -> None:
    ensure_node_tool()
    # Inherits the terminal so the QR code shows up on first login
    subprocess.run(["node", str(STEAM_TOOL / "fetch_demo_urls.js")], check=True)
    download_url_file(DATA / "demo_urls.txt")


def my_steamid() -> str | None:
    account = DATA / "steam_account.json"
    return json.loads(account.read_text())["steamid"] if account.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Fetch demos for your account and build the opening report")
    ap.add_argument("--no-steam", action="store_true", help="skip the Steam download")
    ap.add_argument("--no-replays", action="store_true", help="skip the CS2 replays folder")
    ap.add_argument("--me", default=None, help="your steamid64 (default: from the Steam login)")
    ap.add_argument("--urls", type=Path, help="text file with demo URLs to download (e.g. from the login page)")
    ap.add_argument("--gcpd", type=Path, action="append", default=[],
                    help="saved Steam Personal Game Data page(s) with your match history")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DEMOS.mkdir(parents=True, exist_ok=True)

    if not args.no_steam:
        fetch_from_steam()
    if args.urls:
        download_url_file(args.urls)
    for page in args.gcpd:
        urls = collect.gcpd_demo_urls(page)
        print(f"{len(urls)} replay links in {page.name} (expired ones are skipped)")
        collect.download_urls(urls, DEMOS)
    if not args.no_replays:
        for folder in collect.find_cs2_replays():
            collect.import_dir(folder, DEMOS)

    parse_demo_directory(DEMOS, DATA / "parsed")
    players, teams = engineer.build_dataset(DATA / "parsed")
    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    players.to_parquet(processed / "players.parquet", index=False)
    teams.to_parquet(processed / "teams.parquet", index=False)

    maps = sorted(players["map_name"].unique())
    missing = [m for m in maps if not (DATA / "radar" / f"{m}.png").exists()]
    if missing:
        radar.extract(missing, DATA / "radar")

    out = ROOT / "reports" / "opening_report.html"
    me = args.me or my_steamid()
    sys.argv = ["build_report", "--output", str(out)] + (["--me", me] if me else [])
    build_report.main()
    print(f"\n{players['demo_id'].nunique()} demos, maps: {', '.join(maps)}")


if __name__ == "__main__":
    main()
