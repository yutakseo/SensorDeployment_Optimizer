from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Analysis.internal.visualization import GRID_SIZE_M
from Analysis.internal.placement_animation import savePlacementAnimation

__all__ = ["savePlacementAnimation"]

RESULT_JSON: Final[str] = "/workspace/__RESULTS__/ga/gangjin.full/20-60/20260613_021020.json"
OUTPUT_DIR: Final[str] = "__RESULTS__/animations"
OUTPUT_VIDEO_NAME: Final[str] = "GA_evolution.gif"
FPS: Final[int] = 2
DPI: Final[int] = 100
RADIUS_M: Final[float] = 45.0
FIGURE_SIZE: Final[float] = 5.0
FRAME_STEP: Final[int] = 1
GRID_M: Final[float] = GRID_SIZE_M


def outputPath() -> Path:
    return Path(OUTPUT_DIR) / OUTPUT_VIDEO_NAME


def runPlacementAnimation() -> Path:
    return savePlacementAnimation(
        result_path=RESULT_JSON,
        output_path=outputPath(),
        output_dir=OUTPUT_DIR,
        fps=FPS,
        dpi=DPI,
        radius_m=RADIUS_M,
        size=FIGURE_SIZE,
        step=FRAME_STEP,
        grid_m=GRID_M,
    )


if __name__ == "__main__":
    print(runPlacementAnimation())
