from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Analysis.internal.distance_metrics import Point, asPoints
from Analysis.internal.result_io import loadJson
from Analysis.internal.visualization import GRID_SIZE_M, VisualTool
from Engine.map_loader import MapLoader

DEFAULT_OUTPUT_DIR = "__RESULTS__/analysis/placement_animations"
DEFAULT_FPS = 4
DEFAULT_DPI = 120
DEFAULT_RADIUS_M = 45.0
DEFAULT_SIZE = 6.0
DEFAULT_STEP = 1
CAPTION_HEIGHT_RATIO = 0.14
CAPTION_FONT_SIZE = 11
CAPTION_FINAL_COLOR = "#b91c1c"
CAPTION_NORMAL_COLOR = "#111827"
GIF_EXTENSION = ".gif"
MP4_EXTENSION = ".mp4"
FFMPEG_WRITER = "ffmpeg"
PILLOW_WRITER = "pillow"


@dataclass(frozen=True, slots=True)
class PlacementFrame:
    generation: int
    points: tuple[Point, ...]
    coverage: float | None
    fitness: float | None
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class AnimationConfig:
    result_path: Path
    output_path: Path
    fps: int
    dpi: int
    radius_m: float
    figsize: tuple[float, float]
    step: int
    grid_m: float


def numericValue(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def frameTitle(frame: PlacementFrame, run_name: str) -> str:
    parts = [run_name, f"generation={frame.generation}", f"sensors={len(frame.points)}"]
    if frame.coverage is not None:
        parts.append(f"coverage={frame.coverage:.2f}%")
    return " | ".join(parts)


def frameCaption(frame: PlacementFrame) -> str:
    parts = [f"Generation {frame.generation}", f"Sensors {len(frame.points)}"]
    if frame.coverage is not None:
        parts.append(f"Coverage {frame.coverage:.2f}%")
    else:
        parts.append("Coverage N/A")
    return "   |   ".join(parts)


def generationFrames(run: dict[str, Any], *, step: int) -> list[PlacementFrame]:
    final = run.get("final", {})
    if not isinstance(final, dict):
        final = {}

    corner_points = asPoints(final.get("corner_points", []))
    generations = run.get("generations", [])
    if not isinstance(generations, list):
        generations = []

    frames: list[PlacementFrame] = []
    safe_step = max(1, int(step))
    for index, generation in enumerate(generations):
        if index % safe_step != 0:
            continue
        if not isinstance(generation, dict):
            continue

        generation_number = int(numericValue(generation.get("gen")) or index + 1)
        inner_points = asPoints(generation.get("best_solution", []))
        points = tuple(inner_points + corner_points)
        frames.append(
            PlacementFrame(
                generation=generation_number,
                points=points,
                coverage=numericValue(generation.get("best_coverage")),
                fitness=numericValue(generation.get("best_fitness")),
            )
        )

    final_points = tuple(asPoints(final.get("best_solution", [])) + corner_points)
    if final_points:
        final_generation = frames[-1].generation if frames else len(generations)
        frames.append(
            PlacementFrame(
                generation=final_generation,
                points=final_points,
                coverage=numericValue(final.get("coverage")),
                fitness=numericValue(final.get("fitness")),
                is_final=True,
            )
        )

    if not frames:
        raise ValueError("result JSON has no plottable generation or final placement frames.")
    return frames


def defaultOutputPath(result_path: Path, run: dict[str, Any], output_dir: Path) -> Path:
    run_name = str(run.get("run_name") or result_path.stem)
    map_name = str(run.get("map_name") or "unknown_map").replace(".", "_")
    return output_dir / f"{map_name}_{run_name}_placement_evolution.gif"


def prepareMap(map_name: str, visual: VisualTool, grid_m: float):
    map_data = visual._normalize_image(MapLoader().load(map_name))
    map_data, labels = visual._prepareOverviewMap(map_data, zone_style=None)
    map_data, pad_left, pad_top = visual._padSquareWithOffset(map_data, fill_value=0)
    return map_data, labels, pad_left, pad_top


def drawFrame(
    *,
    axis,
    caption_axis,
    visual: VisualTool,
    map_data,
    labels,
    frame: PlacementFrame,
    run_name: str,
    radius_m: float,
    grid_m: float,
    pad_left: int,
    pad_top: int,
) -> None:
    axis.clear()
    caption_axis.clear()
    shifted_points = visual._shiftPositions(
        list(frame.points),
        x_offset=pad_left,
        y_offset=pad_top,
    )
    visual._drawMapOverview(
        axis,
        map_data,
        title=run_name,
        grid_m=grid_m,
        labels=labels,
    )
    visual._addSensorCircles(
        axis,
        sensor_positions=shifted_points,
        radius_cells=float(radius_m) / float(grid_m),
    )
    caption_axis.set_axis_off()
    caption_axis.set_facecolor("white")
    caption_axis.text(
        0.5,
        0.52,
        frameCaption(frame),
        ha="center",
        va="center",
        fontsize=CAPTION_FONT_SIZE,
        fontweight="bold",
        color=CAPTION_FINAL_COLOR if frame.is_final else CAPTION_NORMAL_COLOR,
        transform=caption_axis.transAxes,
    )


def writerFor(output_path: Path, fps: int):
    extension = output_path.suffix.lower()
    available = set(animation.writers.list())
    if extension == MP4_EXTENSION:
        if FFMPEG_WRITER not in available:
            raise RuntimeError("MP4 output requires matplotlib ffmpeg writer, but it is not available.")
        return animation.FFMpegWriter(fps=fps)
    if extension != GIF_EXTENSION:
        raise ValueError("output extension must be .gif or .mp4")
    if PILLOW_WRITER not in available:
        raise RuntimeError("GIF output requires matplotlib pillow writer, but it is not available.")
    return animation.PillowWriter(fps=fps)


def canvasImage(fig) -> Any:
    from PIL import Image

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = Image.frombuffer(
        "RGBA",
        (width, height),
        fig.canvas.buffer_rgba(),
        "raw",
        "RGBA",
        0,
        1,
    )
    return image.convert("P", palette=Image.Palette.ADAPTIVE).copy()


def saveGif(
    *,
    target: Path,
    fig,
    axis,
    caption_axis,
    frames: Sequence[PlacementFrame],
    visual: VisualTool,
    map_data,
    labels,
    run_name: str,
    radius_m: float,
    grid_m: float,
    pad_left: int,
    pad_top: int,
    fps: int,
) -> None:
    rendered = []
    for frame in frames:
        drawFrame(
            axis=axis,
            caption_axis=caption_axis,
            visual=visual,
            map_data=map_data,
            labels=labels,
            frame=frame,
            run_name=run_name,
            radius_m=radius_m,
            grid_m=grid_m,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        rendered.append(canvasImage(fig))

    duration_ms = int(round(1000 / max(1, int(fps))))
    rendered[0].save(
        target,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def saveMp4(
    *,
    target: Path,
    fig,
    axis,
    caption_axis,
    frames: Sequence[PlacementFrame],
    visual: VisualTool,
    map_data,
    labels,
    run_name: str,
    radius_m: float,
    grid_m: float,
    pad_left: int,
    pad_top: int,
    fps: int,
    dpi: int,
) -> None:
    def update(frame: PlacementFrame):
        drawFrame(
            axis=axis,
            caption_axis=caption_axis,
            visual=visual,
            map_data=map_data,
            labels=labels,
            frame=frame,
            run_name=run_name,
            radius_m=radius_m,
            grid_m=grid_m,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        return axis.patches + axis.lines + axis.texts

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / max(1, int(fps)),
        repeat=True,
        blit=False,
    )
    ani.save(target, writer=writerFor(target, int(fps)), dpi=int(dpi))


def savePlacementAnimation(
    *,
    result_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    radius_m: float = DEFAULT_RADIUS_M,
    size: float = DEFAULT_SIZE,
    step: int = DEFAULT_STEP,
    grid_m: float = GRID_SIZE_M,
) -> Path:
    source = Path(result_path)
    run = loadJson(source)
    map_name = str(run.get("map_name") or run.get("meta", {}).get("map_name") or "")
    if not map_name:
        raise ValueError("result JSON does not contain map_name.")

    target = Path(output_path) if output_path is not None else defaultOutputPath(
        source,
        run,
        Path(output_dir),
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    frames = generationFrames(run, step=step)
    run_name = str(run.get("run_name") or source.stem)
    visual = VisualTool(show=False, save=False, size=(size, size), dpi=dpi, save_title=True)
    map_data, labels, pad_left, pad_top = prepareMap(map_name, visual, grid_m)

    fig = plt.figure(figsize=(float(size), float(size)), dpi=int(dpi))
    axis = fig.add_axes([0.0, CAPTION_HEIGHT_RATIO, 1.0, 1.0 - CAPTION_HEIGHT_RATIO])
    caption_axis = fig.add_axes([0.0, 0.0, 1.0, CAPTION_HEIGHT_RATIO])

    if target.suffix.lower() == GIF_EXTENSION:
        saveGif(
            target=target,
            fig=fig,
            axis=axis,
            caption_axis=caption_axis,
            frames=frames,
            visual=visual,
            map_data=map_data,
            labels=labels,
            run_name=run_name,
            radius_m=radius_m,
            grid_m=grid_m,
            pad_left=pad_left,
            pad_top=pad_top,
            fps=int(fps),
        )
    else:
        saveMp4(
            target=target,
            fig=fig,
            axis=axis,
            caption_axis=caption_axis,
            frames=frames,
            visual=visual,
            map_data=map_data,
            labels=labels,
            run_name=run_name,
            radius_m=radius_m,
            grid_m=grid_m,
            pad_left=pad_left,
            pad_top=pad_top,
            fps=int(fps),
            dpi=int(dpi),
        )
    plt.close(fig)
    return target


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sensor-placement evolution animation from one optimizer result JSON."
    )
    parser.add_argument("result_json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--grid-m", type=float, default=GRID_SIZE_M)
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    output_path = savePlacementAnimation(
        result_path=args.result_json,
        output_path=args.output,
        output_dir=args.output_dir,
        fps=args.fps,
        dpi=args.dpi,
        radius_m=args.radius_m,
        size=args.size,
        step=args.step,
        grid_m=args.grid_m,
    )
    print(output_path)


if __name__ == "__main__":
    main()
