"""
Steam Web API integration for CS2 match history and demo downloads.

Fetches recent match data and demo download URLs via Steam's API.
Uses aiohttp for async HTTP. Config from env vars or .env file.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

STEAM_API_BASE = "https://api.steampowered.com"
CS2_APP_ID = 730  # CS2 uses the same app ID as CS:GO

# Valve's match history endpoint (requires Game Coordinator access)
# For public match sharing codes, we use the unofficial but widely-used endpoint
MATCH_HISTORY_URL = f"{STEAM_API_BASE}/ICSGOPlayers_730/GetNextMatchSharingCode/v1"
MATCH_LIST_URL = f"{STEAM_API_BASE}/ICSGOPlayers_730/GetRecentGames/v1"


class SteamConfig:
    """Steam API configuration from environment."""

    def __init__(self):
        self.api_key: str = os.environ.get("STEAM_API_KEY", "")
        self.steam_id: str = os.environ.get("STEAM_ID", "")
        self.auth_code: str = os.environ.get("STEAM_AUTH_CODE", "")
        self.download_dir: Path = Path(
            os.environ.get("DEMO_DOWNLOAD_DIR", "data/demos")
        )
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        if not self.api_key:
            logger.error("STEAM_API_KEY not set")
            return False
        if not self.steam_id:
            logger.error("STEAM_ID not set")
            return False
        return True


class SteamAPI:
    """Async Steam Web API client for CS2 match data."""

    def __init__(self, config: Optional[SteamConfig] = None):
        self.config = config or SteamConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _api_get(self, url: str, params: dict) -> dict:
        """Make authenticated GET request to Steam API."""
        params["key"] = self.config.api_key
        session = await self._get_session()

        async with session.get(url, params=params) as resp:
            if resp.status == 403:
                raise PermissionError("Invalid Steam API key or insufficient permissions")
            if resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited, waiting %ds", retry_after)
                await asyncio.sleep(retry_after)
                return await self._api_get(url, params)
            resp.raise_for_status()
            return await resp.json()

    async def get_next_match_sharing_code(
        self,
        steam_id: Optional[str] = None,
        known_code: str = "CSGO-00000-00000-00000-00000-00000",
    ) -> Optional[str]:
        """
        Get the next match sharing code after a known code.

        Uses Valve's GetNextMatchSharingCode endpoint.
        Requires the player's auth code (from game settings).
        """
        params = {
            "steamid": steam_id or self.config.steam_id,
            "steamidkey": self.config.auth_code,
            "knowncode": known_code,
        }

        try:
            data = await self._api_get(MATCH_HISTORY_URL, params)
            result = data.get("result", {})
            next_code = result.get("nextcode")
            if next_code == "n/a":
                logger.info("No new matches available")
                return None
            return next_code
        except Exception as e:
            logger.error("Failed to get match sharing code: %s", e)
            return None

    async def get_match_details(self, match_id: str) -> Optional[dict]:
        """
        Fetch match details including demo download URL.

        Note: Valve's official API for full match details requires
        the Game Coordinator protocol. This uses the web API where available.
        For full GCPD access, consider using csgo library or similar.
        """
        url = f"{STEAM_API_BASE}/ICSGOPlayers_730/GetMatchDetails/v1"
        params = {"matchid": match_id}

        try:
            data = await self._api_get(url, params)
            return data.get("result")
        except Exception as e:
            logger.error("Failed to get match details for %s: %s", match_id, e)
            return None

    async def get_player_stats(
        self, steam_id: Optional[str] = None
    ) -> Optional[dict]:
        """Fetch CS2 player stats (kills, deaths, wins, etc.)."""
        url = f"{STEAM_API_BASE}/ISteamUserStats/GetUserStatsForGame/v2"
        params = {
            "steamid": steam_id or self.config.steam_id,
            "appid": CS2_APP_ID,
        }

        try:
            data = await self._api_get(url, params)
            stats = data.get("playerstats", {}).get("stats", [])
            return {s["name"]: s["value"] for s in stats}
        except Exception as e:
            logger.error("Failed to get player stats: %s", e)
            return None

    async def download_demo(
        self,
        demo_url: str,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download a demo file from Valve's CDN.

        Demo URLs are typically: http://replay{N}.valve.net/730/{match_id}_{token}.dem.bz2
        """
        if not demo_url:
            logger.error("Empty demo URL")
            return None

        if filename is None:
            filename = demo_url.split("/")[-1]
        output_path = self.config.download_dir / filename

        if output_path.exists():
            logger.info("Demo already downloaded: %s", output_path)
            return output_path

        logger.info("Downloading demo: %s", demo_url)
        session = await self._get_session()

        try:
            async with session.get(demo_url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))

                with open(output_path, "wb") as f:
                    downloaded = 0
                    async for chunk in resp.content.iter_chunked(1024 * 64):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (1024 * 1024) == 0:
                            pct = (downloaded / total) * 100
                            logger.info("Download progress: %.1f%%", pct)

            logger.info("Downloaded demo to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
            return output_path

        except Exception as e:
            logger.error("Failed to download demo: %s", e)
            if output_path.exists():
                output_path.unlink()
            return None

    async def collect_recent_demos(
        self,
        max_matches: int = 10,
    ) -> list[Path]:
        """
        Iterate through recent match sharing codes and download demos.

        Returns list of downloaded demo file paths.
        """
        if not self.config.validate():
            raise ValueError("Steam config not properly set. Check env vars.")

        downloaded = []
        code = "CSGO-00000-00000-00000-00000-00000"

        for i in range(max_matches):
            next_code = await self.get_next_match_sharing_code(known_code=code)
            if not next_code:
                logger.info("No more matches after %d codes", i)
                break

            code = next_code
            logger.info("Match %d: sharing code = %s", i + 1, code)

            # Get match details for demo URL
            details = await self.get_match_details(code)
            if details and "demo_url" in details:
                path = await self.download_demo(details["demo_url"])
                if path:
                    downloaded.append(path)

        return downloaded


async def main():
    """CLI entry point: download recent demos."""
    logging.basicConfig(level=logging.INFO)

    async with SteamAPI() as api:
        stats = await api.get_player_stats()
        if stats:
            print(f"Player stats: {len(stats)} entries")
            for key in ["total_kills", "total_deaths", "total_wins"]:
                if key in stats:
                    print(f"  {key}: {stats[key]}")

        demos = await api.collect_recent_demos(max_matches=5)
        print(f"Downloaded {len(demos)} demos")


if __name__ == "__main__":
    asyncio.run(main())
