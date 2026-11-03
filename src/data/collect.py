"""
Gather demo files into data/raw/demos.

    python -m src.data.collect --cs2-replays          # demos you downloaded in CS2 (Watch > Your Matches)
    python -m src.data.collect --from-dir PATH        # any folder: .dem, .dem.bz2, .zip (e.g. HLTV downloads)
    python -m src.data.collect --urls urls.txt        # direct links to .dem / .dem.bz2 files
    python -m src.data.collect --gcpd page.html       # saved Steam "Personal Game Data" match history page

HLTV ships demos as .rar; extract those first, then use --from-dir.
"""

import argparse
import bz2
import logging
import re
import shutil
import winreg
import zipfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CS2_REPLAYS = Path("steamapps/common/Counter-Strike Global Offensive/game/csgo/replays")
MATCH_ID = re.compile(r"(\d{15,})_(\d+)")
REPLAY_URL = re.compile(r"https?://replay\d+\.valve\.net/730/\d+_\d+\.dem\.bz2")


def demo_key(name: str) -> str:
    """Same match, same key: 'match730_<id>_<token>_154.dem' (CS2) and '<id>_<token>.dem.bz2' (Valve CDN)."""
    m = MATCH_ID.search(name)
    return f"{int(m.group(1))}_{m.group(2)}" if m else name


def already_have(dest_dir: Path, name: str) -> bool:
    key = demo_key(name)
    return any(demo_key(f.name) == key for f in dest_dir.glob("*.dem"))


def steam_libraries() -> list[Path]:
    """Steam install dir plus every library folder listed in libraryfolders.vdf."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            steam = Path(winreg.QueryValueEx(key, "SteamPath")[0])
    except OSError:
        steam = Path("C:/Program Files (x86)/Steam")
    libs = [steam]
    vdf = steam / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        for path in re.findall(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore")):
            libs.append(Path(path.replace("\\\\", "\\")))
    return list(dict.fromkeys(libs))


def find_cs2_replays() -> list[Path]:
    return [lib / CS2_REPLAYS for lib in steam_libraries() if (lib / CS2_REPLAYS).is_dir()]


def import_file(src: Path, dest_dir: Path) -> Path | None:
    """Copy or unpack one demo into dest_dir. Existing demos are left alone."""
    name = src.name
    if name.endswith(".dem"):
        dest = dest_dir / name
        if not dest.exists() and not already_have(dest_dir, name):
            shutil.copy2(src, dest)
        return dest
    if name.endswith(".dem.bz2"):
        dest = dest_dir / name[:-4]
        if not dest.exists() and not already_have(dest_dir, name):
            with bz2.open(src) as fin, open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        return dest
    if name.endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            for member in zf.namelist():
                if member.endswith(".dem"):
                    dest = dest_dir / f"{src.stem}_{Path(member).name}"
                    if not dest.exists():
                        with zf.open(member) as fin, open(dest, "wb") as fout:
                            shutil.copyfileobj(fin, fout)
        return dest_dir
    return None


def import_dir(src_dir: Path, dest_dir: Path) -> int:
    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and import_file(f, dest_dir):
            count += 1
    logger.info("Imported %d files from %s", count, src_dir)
    return count


def download_urls(urls: list[str], dest_dir: Path) -> int:
    count = 0
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        for url in urls:
            tmp = dest_dir / url.rstrip("/").split("/")[-1].split("?")[0]
            if not tmp.name.endswith((".dem", ".dem.bz2")):
                logger.warning("Skip (not a .dem/.dem.bz2 link): %s", url)
                continue
            if already_have(dest_dir, tmp.name):
                continue
            if not tmp.exists():
                logger.info("Downloading %s", url)
                with client.stream("GET", url) as resp:
                    if resp.status_code == 404:
                        logger.warning("Expired on Valve's servers (404): %s", url)
                        continue
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes(1 << 16):
                            f.write(chunk)
            if tmp.name.endswith(".bz2") and tmp.exists():
                import_file(tmp, dest_dir)
                tmp.unlink()
            count += 1
    return count


def gcpd_demo_urls(html_path: Path) -> list[str]:
    """Replay links from a saved steamcommunity.com/my/gcpd/730 match history page."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    return list(dict.fromkeys(REPLAY_URL.findall(html)))


def main():
    ap = argparse.ArgumentParser(description="Collect CS2 demos into one folder")
    ap.add_argument("--cs2-replays", action="store_true", help="import demos from your CS2 replays folder")
    ap.add_argument("--from-dir", type=Path)
    ap.add_argument("--urls", type=Path, help="text file with one demo URL per line")
    ap.add_argument("--gcpd", type=Path, help="saved Steam Personal Game Data page (.html)")
    ap.add_argument("--output", type=Path, default=Path("data/raw/demos"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.cs2_replays:
        folders = find_cs2_replays()
        if not folders:
            raise SystemExit("CS2 replays folder not found. Is CS2 installed?")
        for folder in folders:
            import_dir(folder, args.output)
    if args.from_dir:
        import_dir(args.from_dir, args.output)
    if args.urls:
        urls = [u.strip() for u in args.urls.read_text().splitlines() if u.strip()]
        download_urls(urls, args.output)
    if args.gcpd:
        urls = gcpd_demo_urls(args.gcpd)
        print(f"{len(urls)} replay links found in {args.gcpd}")
        download_urls(urls, args.output)
    if not (args.cs2_replays or args.from_dir or args.urls or args.gcpd):
        ap.print_help()
        return
    print(f"{len(list(args.output.glob('*.dem')))} demos in {args.output}")


if __name__ == "__main__":
    main()
