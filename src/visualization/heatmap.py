"""Generate position heatmaps overlaid on CS2 radar map images."""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from PIL import Image
except ImportError:
    plt = None


def create_heatmap(
    positions: pd.DataFrame,
    map_name: str = "de_dust2",
    radar_path: str | Path | None = None,
    output_path: str | Path = "heatmap.png",
    title: str = "Player Positions",
    bins: int = 64,
) -> None:
    """Create a 2D heatmap of player positions.

    Args:
        positions: DataFrame with 'X' and 'Y' columns
        map_name: Map name for title
        radar_path: Optional path to radar background image
        output_path: Where to save the heatmap
        title: Plot title
        bins: Histogram bin count
    """
    if plt is None:
        raise ImportError("matplotlib required: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 10))

    if radar_path and Path(radar_path).exists():
        bg = Image.open(radar_path)
        ax.imshow(bg, extent=[positions["X"].min(), positions["X"].max(),
                              positions["Y"].min(), positions["Y"].max()],
                  alpha=0.5)

    heatmap, xedges, yedges = np.histogram2d(
        positions["X"], positions["Y"], bins=bins
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(
        heatmap.T, extent=extent, origin="lower",
        cmap="hot", alpha=0.7, interpolation="gaussian",
    )
    ax.set_title(f"{title} — {map_name}", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    print("Heatmap generator — requires parsed position data")
