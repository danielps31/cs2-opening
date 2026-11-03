"""
Extract radar images, overview coordinates and weapon icons from your CS2 install.

    python -m src.data.radar [--maps de_dust2 de_inferno ...]
    python -m src.data.radar --icons

Reads panorama/images/overheadmaps/<map>_radar_psd.vtex_c and
resource/overviews/<map>.txt from pak01_dir.vpk, writes data/radar/<map>.png
and data/radar/overviews.json ({map: {pos_x, pos_y, scale}}).
"""

import argparse
import json
import re
import struct
from pathlib import Path

import lz4.block
import vpk
from PIL import Image

from src.data.collect import steam_libraries

CS2_VPK = Path("steamapps/common/Counter-Strike Global Offensive/game/csgo/pak01_dir.vpk")

# Source 2 VTEX formats we can decode -> (bytes per pixel or None for block formats, Pillow decoder args)
FORMATS = {
    1: ("bcn", 1),          # DXT1
    2: ("bcn", 3),          # DXT5
    4: ("raw", "RGBA"),     # RGBA8888
    20: ("bcn", 7),         # BC7
    28: ("raw", "BGRA"),    # BGRA8888
}
EXTRA_COMPRESSED_MIP_SIZE = 4


def find_vpk() -> Path:
    for lib in steam_libraries():
        if (lib / CS2_VPK).exists():
            return lib / CS2_VPK
    raise FileNotFoundError("CS2 pak01_dir.vpk not found. Is CS2 installed?")


def mip_size(fmt: int, w: int, h: int) -> int:
    if FORMATS[fmt][0] == "raw":
        return w * h * 4
    block = 8 if fmt == 1 else 16
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block


def decode_vtex(data: bytes) -> Image.Image:
    """Largest mip of a .vtex_c texture as an RGBA image."""
    file_size, _, _, block_offset, block_count = struct.unpack_from("<IHHII", data, 0)
    blocks = {}
    for i in range(block_count):
        pos = 8 + block_offset + 12 * i
        offset, size = struct.unpack_from("<II", data, pos + 4)
        blocks[data[pos:pos + 4]] = pos + 4 + offset
    d = blocks[b"DATA"]
    w, h = struct.unpack_from("<HH", data, d + 20)
    fmt, mips = data[d + 26], data[d + 27]
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported texture format {fmt}")

    compressed = None
    extra_pos = d + 32
    extra_offset, extra_count = struct.unpack_from("<II", data, extra_pos)
    for i in range(extra_count):
        entry = extra_pos + extra_offset + 12 * i
        typ, off, _ = struct.unpack_from("<III", data, entry)
        if typ == EXTRA_COMPRESSED_MIP_SIZE:
            start = entry + 4 + off
            is_compressed, mips_offset, n = struct.unpack_from("<III", data, start)
            sizes_at = start + 4 + mips_offset
            if is_compressed:
                compressed = list(struct.unpack_from(f"<{n}i", data, sizes_at))

    # Mips are stored smallest first; the full-size one is last
    pos = file_size
    for level in range(mips - 1, -1, -1):
        lw, lh = max(1, w >> level), max(1, h >> level)
        raw_size = mip_size(fmt, lw, lh)
        if compressed:
            chunk = data[pos:pos + compressed[level]]
            pos += compressed[level]
            pixels = lz4.block.decompress(chunk, uncompressed_size=raw_size) if len(chunk) < raw_size else chunk
        else:
            pixels = data[pos:pos + raw_size]
            pos += raw_size
    decoder, arg = FORMATS[fmt]
    return Image.frombytes("RGBA", (w, h), pixels, decoder, arg)


def parse_overview(text: str) -> dict:
    values = dict(re.findall(r'"(\w+)"\s+"([^"]*)"', text))
    return {k: float(values[k]) for k in ("pos_x", "pos_y", "scale")}


def extract(maps: list[str] | None, out_dir: Path) -> dict:
    pak = vpk.open(str(find_vpk()))
    out_dir.mkdir(parents=True, exist_ok=True)
    overviews_path = out_dir / "overviews.json"
    overviews = json.loads(overviews_path.read_text()) if overviews_path.exists() else {}
    available = sorted(
        m.group(1) for name in pak
        if (m := re.match(r"panorama/images/overheadmaps/(de_\w+?)_radar_psd\.vtex_c$", name))
    )
    for map_name in maps or available:
        try:
            img = decode_vtex(pak[f"panorama/images/overheadmaps/{map_name}_radar_psd.vtex_c"].read())
            img.save(out_dir / f"{map_name}.png")
            overviews[map_name] = parse_overview(pak[f"resource/overviews/{map_name}.txt"].read().decode("utf-8", "ignore"))
            print(f"{map_name}: {img.size[0]}x{img.size[1]}")
        except (KeyError, ValueError) as e:
            print(f"{map_name}: skipped ({e})")
    overviews_path.write_text(json.dumps(overviews, indent=2))
    return overviews


def extract_icons(out_dir: Path) -> int:
    """Weapon/equipment icons (panorama/images/icons/equipment/*.vsvg_c) as plain .svg files."""
    pak = vpk.open(str(find_vpk()))
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in pak:
        m = re.match(r"panorama/images/icons/equipment/(\w+)\.vsvg_c$", name)
        if not m:
            continue
        data = pak[name].read()
        start, end = data.find(b"<svg"), data.rfind(b"</svg>")
        if start < 0 or end < 0:
            continue
        (out_dir / f"{m.group(1)}.svg").write_bytes(data[start:end + len(b"</svg>")])
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Extract CS2 radar images from the game files")
    ap.add_argument("--maps", nargs="*", help="e.g. de_dust2 de_mirage (default: every de_ map)")
    ap.add_argument("--output", type=Path, default=Path("data/radar"))
    ap.add_argument("--icons", action="store_true", help="extract weapon icons to data/icons instead")
    args = ap.parse_args()
    if args.icons:
        print(f"{extract_icons(Path('data/icons'))} icons -> data/icons")
    else:
        extract(args.maps, args.output)


if __name__ == "__main__":
    main()
