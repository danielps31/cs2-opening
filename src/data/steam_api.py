"""
Steam Web API client for CS2 match sharecodes.

What Steam gives you over the Web API:
  - your match history as sharecodes (CSGO-xxxxx-...), one at a time, via
    ICSGOPlayers_730/GetNextMatchSharingCode. This needs YOUR auth code
    (https://help.steampowered.com/en/wizard/HelpWithGameIssue/?appid=730&issueid=128),
    so it only works for your own account or accounts that share their code.

What it does NOT give you: demo files. Turning a sharecode into a replay URL
needs the CS2 Game Coordinator (a logged-in Steam client). Until that's wired
up, download demos in-game (Watch > Your Matches) and use
`python -m src.data.collect --cs2-replays`.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

STEAM_API_BASE = "https://api.steampowered.com"
NEXT_SHARECODE_URL = f"{STEAM_API_BASE}/ICSGOPlayers_730/GetNextMatchSharingCode/v1"
MAX_RETRIES = 3

SHARECODE_ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZabcdefhijkmnopqrstuvwxyz23456789"


def decode_sharecode(code: str) -> dict:
    """Sharecode -> match id, outcome (reservation) id and TV port, as the Game Coordinator expects."""
    chars = code.removeprefix("CSGO-").replace("-", "")
    n = 0
    for c in reversed(chars):
        n = n * len(SHARECODE_ALPHABET) + SHARECODE_ALPHABET.index(c)
    raw = n.to_bytes(18, "big")
    return {
        "match_id": int.from_bytes(raw[0:8], "little"),
        "outcome_id": int.from_bytes(raw[8:16], "little"),
        "token": int.from_bytes(raw[16:18], "little"),
    }


class SteamAPI:
    def __init__(self, api_key: str | None = None, steam_id: str | None = None, auth_code: str | None = None):
        self.api_key = api_key or os.environ.get("STEAM_API_KEY", "")
        self.steam_id = steam_id or os.environ.get("STEAM_ID", "")
        self.auth_code = auth_code or os.environ.get("STEAM_AUTH_CODE", "")
        self.client = httpx.Client(timeout=30)

    def _get(self, url: str, params: dict) -> httpx.Response:
        params = {**params, "key": self.api_key}
        for attempt in range(MAX_RETRIES):
            resp = self.client.get(url, params=params)
            if resp.status_code != 429:
                return resp
            wait = int(resp.headers.get("Retry-After", 5))
            logger.warning("Rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
        return resp

    def next_sharecode(self, known_code: str) -> str | None:
        """Sharecode of the match after known_code, or None when there's no newer match."""
        resp = self._get(NEXT_SHARECODE_URL, {
            "steamid": self.steam_id, "steamidkey": self.auth_code, "knowncode": known_code,
        })
        if resp.status_code == 202:
            return None  # no newer match yet
        if resp.status_code == 403:
            raise PermissionError("Steam rejected the request: check STEAM_API_KEY / STEAM_AUTH_CODE")
        if resp.status_code == 412:
            raise ValueError("Steam does not accept this known code; use a recent sharecode of your own match")
        resp.raise_for_status()
        code = resp.json().get("result", {}).get("nextcode")
        return None if code in (None, "n/a") else code

    def sharecodes_after(self, known_code: str, limit: int = 50) -> list[str]:
        codes = []
        code = known_code
        while len(codes) < limit:
            code = self.next_sharecode(code)
            if not code:
                break
            codes.append(code)
        return codes


def main():
    ap = argparse.ArgumentParser(description="Fetch your CS2 match sharecodes from Steam")
    ap.add_argument("--known-code", required=True,
                    help="a sharecode of one of your matches (CS2 > Watch > Your Matches > copy share link)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--output", default="data/sharecodes.txt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    api = SteamAPI()
    if not (api.api_key and api.steam_id and api.auth_code):
        raise SystemExit("Set STEAM_API_KEY, STEAM_ID and STEAM_AUTH_CODE in .env")
    codes = api.sharecodes_after(args.known_code, args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = set(out.read_text().split()) if out.exists() else set()
    out.write_text("\n".join(sorted(existing | set(codes))) + "\n")
    print(f"{len(codes)} new sharecodes -> {out}")


if __name__ == "__main__":
    main()
