from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt


PathLike = Union[str, Path]


@dataclass
class Analyzer:
    """
    Analyzer supports two construction modes:

    1) Backward-compatible (existing):
       analyzer = Analyzer(run_data=<dict>)

    2) Path-based (new):
       analyzer = Analyzer(
           result_root_path="/workspace/__RESULTS__/gangjin.crop2",
           file_path=7                # index
       )
       analyzer = Analyzer(result_root_path="...", file_path="0007.json")   # relative
       analyzer = Analyzer(result_root_path="...", file_path="0007")        # stem
       analyzer = Analyzer(result_root_path="...", file_path="/abs/.../0007.json")  # absolute
    """

    # ---- data ----
    run: Dict[str, Any] = field(default_factory=dict, init=False)

    # ---- ctor inputs (optional) ----
    run_data: Optional[Dict[str, Any]] = None
    result_root_path: Optional[PathLike] = None
    file_path: Optional[Union[int, PathLike]] = None

    # ---- internal ----
    _loaded_from: Optional[str] = field(default=None, init=False)
    _json_files_cache: Optional[List[Path]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Priority:
        1) If run_data provided -> use it (backward compatible)
        2) Else if result_root_path + file_path provided -> resolve and load json
        3) Else -> error
        """
        if self.run_data is not None:
            if not isinstance(self.run_data, dict) or len(self.run_data) == 0:
                raise ValueError("run_data must be a non-empty dict.")
            self.run = self.run_data
            self._loaded_from = "run_data(dict)"
            return

        if self.result_root_path is None or self.file_path is None:
            raise ValueError(
                "Analyzer requires either:\n"
                "  - run_data=<dict>\n"
                "or\n"
                "  - result_root_path=<dir> and file_path=<index|path>\n"
            )

        root = Path(self.result_root_path)
        if not root.exists():
            raise FileNotFoundError(f"result_root_path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"result_root_path must be a directory: {root}")

        json_path = self._resolve_json_path(root, self.file_path)
        with open(json_path, "r", encoding="utf-8") as f:
            self.run = json.load(f)
        self._loaded_from = str(json_path)

    @property
    def loaded_from(self) -> Optional[str]:
        return self._loaded_from

    # =========================
    # Path resolving utilities
    # =========================
    def _list_json_files(self, root: Path) -> List[Path]:
        if self._json_files_cache is not None:
            return self._json_files_cache

        files = list(root.glob("*.json"))

        def sort_key(p: Path):
            s = p.stem
            if s.isdigit():
                return (0, int(s))
            return (1, s)

        self._json_files_cache = sorted(files, key=sort_key)
        return self._json_files_cache

    def _resolve_json_path(self, root: Path, file_path: Union[int, PathLike]) -> Path:
        # 1) index mode
        if isinstance(file_path, int):
            files = self._list_json_files(root)
            if len(files) == 0:
                raise FileNotFoundError(f"No .json files under: {root}")
            if file_path < 0 or file_path >= len(files):
                raise IndexError(f"index={file_path} out of range (0 ~ {len(files)-1})")
            return files[file_path]

        # 2) path mode
        p = Path(file_path)

        # absolute path
        cand = p if p.is_absolute() else (root / p)

        # exact match
        if cand.exists() and cand.is_file():
            return cand

        # extension omitted -> try .json
        if cand.suffix == "":
            cand_json = cand.with_suffix(".json")
            if cand_json.exists() and cand_json.is_file():
                return cand_json

        # stem match anywhere under root (common convenience)
        stem = Path(file_path).stem
        matches = [f for f in self._list_json_files(root) if f.stem == stem]
        if len(matches) == 1:
            return matches[0]

        # if digits, interpret as index (optional convenience)
        if stem.isdigit():
            idx = int(stem)
            files = self._list_json_files(root)
            if 0 <= idx < len(files):
                return files[idx]

        raise FileNotFoundError(
            "Could not resolve json file.\n"
            f"- result_root_path: {root}\n"
            f"- file_path: {file_path}\n"
            "Accepted file_path:\n"
            "  - int index (e.g., 7)\n"
            "  - relative path under result_root_path (e.g., '0007.json')\n"
            "  - stem without extension (e.g., '0007')\n"
            "  - absolute path (e.g., '/abs/.../0007.json')\n"
        )

    # =========================
    # Plot helpers
    # =========================
    def _apply_xticks(self, x: List[int], *, xtick_step: int) -> None:
        """
        xtick_step:
          - 1: 모든 세대 표시
          - 5: 5세대마다 표시
        마지막 세대는 항상 tick에 포함.
        """
        if not isinstance(xtick_step, int) or xtick_step <= 0:
            raise ValueError(f"xtick_step must be a positive int. Got: {xtick_step}")

        if len(x) == 0:
            return

        ticks = x[::xtick_step]
        if len(ticks) == 0 or ticks[-1] != x[-1]:
            ticks = list(ticks) + [x[-1]]

        plt.xticks(ticks)

    def _resolve_save_path(
        self,
        *,
        save_path: Optional[PathLike],
        save_dir: Optional[PathLike],
        default_stem: str,
        ext: str,
    ) -> Optional[Path]:
        """
        Priority:
          1) save_path (file) -> exact
          2) save_dir (dir) + auto filename
          3) None
        """
        if save_path is None and save_dir is None:
            return None

        if save_path is not None:
            p = Path(save_path)
            # if user passes a directory accidentally, drop file inside
            if p.exists() and p.is_dir():
                run_name = str(self.run.get("run_name", "unknown_run"))
                return p / f"{run_name}_{default_stem}.{ext}"
            # if no suffix, append ext
            if p.suffix == "":
                return p.with_suffix(f".{ext}")
            return p

        # save_dir mode
        d = Path(save_dir)
        run_name = str(self.run.get("run_name", "unknown_run"))
        return d / f"{run_name}_{default_stem}.{ext}"

    def _maybe_savefig(
        self,
        *,
        save_path: Optional[PathLike],
        save_dir: Optional[PathLike],
        default_stem: str,
        ext: str = "png",
        save_dpi: Optional[int] = None,
        overwrite: bool = True,
    ) -> Optional[Path]:
        out = self._resolve_save_path(
            save_path=save_path,
            save_dir=save_dir,
            default_stem=default_stem,
            ext=ext,
        )
        if out is None:
            return None

        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not overwrite:
            raise FileExistsError(f"save_path exists (overwrite=False): {out}")

        plt.savefig(out, dpi=(save_dpi if save_dpi is not None else None), bbox_inches="tight")
        return out

    # =========================
    # Existing analysis methods
    # =========================
    def _get_generations(self) -> List[Dict[str, Any]]:
        gens = self.run.get("generations", [])
        if not isinstance(gens, list) or len(gens) == 0:
            raise ValueError("run['generations'] is empty or not a list.")
        return gens

    def _default_corner_count(self) -> int:
        final = self.run.get("final", {}) or {}
        corner_points = final.get("corner_points", []) or []
        return len(corner_points)

    def _best_inner_count(self, g: Dict[str, Any]) -> int:
        return len(g.get("best_solution", []) or [])

    def _best_corner_count(self, g: Dict[str, Any], fallback_corner_count: int) -> int:
        for k in ("best_corner_points", "corner_solution", "corner_points"):
            if k in g and isinstance(g[k], list):
                return len(g[k])
        return fallback_corner_count

    def plot_evolution_trend(
        self,
        *,
        include_corners: bool = True,
        corner_count: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10.0, 4.8),
        dpi: int = 100,
        show: bool = True,
        xtick_step: int = 1,
        annotate_last: bool = True,
        # ---- saving ----
        save_path: Optional[PathLike] = None,
        save_dir: Optional[PathLike] = None,
        save_ext: str = "png",
        save_dpi: Optional[int] = None,
        overwrite: bool = True,
        close: bool = False,
    ) -> None:
        gens = self._get_generations()

        x = [int(g["gen"]) for g in gens]
        smin = [float(g["sensors_min"]) for g in gens]
        smax = [float(g["sensors_max"]) for g in gens]

        fallback_corner_count = self._default_corner_count() if corner_count is None else int(corner_count)

        best_counts: List[int] = []
        for g in gens:
            inner = self._best_inner_count(g)
            if include_corners:
                corners = self._best_corner_count(g, fallback_corner_count)
                best_counts.append(inner + corners)
            else:
                best_counts.append(inner)

        plt.figure(figsize=figsize, dpi=dpi)

        plt.fill_between(
            x,
            smin,
            smax,
            alpha=0.35,
            color="skyblue",
            label="Sensor count range (min–max)",
        )

        plt.plot(
            x,
            best_counts,
            marker="o",
            linewidth=2,
            label=("Best sensor count (inner+corner)" if include_corners else "Best sensor count (inner only)"),
        )

        # 마지막 결과 센서수 점 위에 라벨
        if len(x) > 0:
            last_x = x[-1]
            last_y = best_counts[-1]
            plt.scatter([last_x], [last_y], zorder=5)
            if annotate_last:
                plt.annotate(
                    f"{int(last_y)}",
                    (last_x, last_y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=10,
                )

        if title is None:
            run_name = self.run.get("run_name", "unknown_run")
            title = f"GA evolution trend: sensor count vs generation ({run_name})"

        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Number of sensors")

        self._apply_xticks(x, xtick_step=xtick_step)

        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        # 저장
        self._maybe_savefig(
            save_path=save_path,
            save_dir=save_dir,
            default_stem="evolution_trend",
            ext=save_ext,
            save_dpi=save_dpi,
            overwrite=overwrite,
        )

        if show:
            plt.show()

        if close:
            plt.close()

        return None  # 기존 반환 유지

    def _extract_best_coverage(self, g: Dict[str, Any]) -> float:
        if "best_coverage" in g and g["best_coverage"] is not None:
            return float(g["best_coverage"])
        if "best_fitness" in g and g["best_fitness"] is not None:
            return float(g["best_fitness"])
        raise KeyError("Generation item has neither 'best_coverage' nor 'best_fitness'.")

    def plot_coverage_trend(
        self,
        *,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10.0, 4.0),
        dpi: int = 100,
        show: bool = True,
        ylim: Optional[Tuple[float, float]] = (0.0, 100.0),
        marker: str = "o",
        linewidth: float = 2.0,
        xtick_step: int = 1,
        # ---- saving ----
        save_path: Optional[PathLike] = None,
        save_dir: Optional[PathLike] = None,
        save_ext: str = "png",
        save_dpi: Optional[int] = None,
        overwrite: bool = True,
        close: bool = False,
    ) -> None:
        gens = self._get_generations()

        x = [int(g["gen"]) for g in gens]
        best_cov = [self._extract_best_coverage(g) for g in gens]

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(x, best_cov, marker=marker, linewidth=linewidth, label="Best coverage")

        if title is None:
            run_name = self.run.get("run_name", "unknown_run")
            title = f"Coverage Evolution over Generations ({run_name})"

        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Coverage (%)")

        self._apply_xticks(x, xtick_step=xtick_step)

        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if ylim is not None:
            plt.ylim(ylim)

        # 저장
        self._maybe_savefig(
            save_path=save_path,
            save_dir=save_dir,
            default_stem="coverage_trend",
            ext=save_ext,
            save_dpi=save_dpi,
            overwrite=overwrite,
        )

        if show:
            plt.show()

        if close:
            plt.close()

        return None  # 기존 반환 유지

    def plot_fitness_trend(
        self,
        *,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10.0, 4.8),
        dpi: int = 100,
        show: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        plot_avg: bool = True,
        plot_best: bool = True,
        xtick_step: int = 1,
        # ---- saving ----
        save_path: Optional[PathLike] = None,
        save_dir: Optional[PathLike] = None,
        save_ext: str = "png",
        save_dpi: Optional[int] = None,
        overwrite: bool = True,
        close: bool = False,
    ) -> None:
        gens = self._get_generations()

        x = [int(g["gen"]) for g in gens]
        fmin = [float(g["fitness_min"]) for g in gens]
        fmax = [float(g["fitness_max"]) for g in gens]
        favg = [float(g["fitness_avg"]) for g in gens]
        fbest = [float(g.get("best_fitness", g["fitness_max"])) for g in gens]

        plt.figure(figsize=figsize, dpi=dpi)

        plt.fill_between(
            x,
            fmin,
            fmax,
            alpha=0.25,
            label="Fitness range (min–max)",
        )

        if plot_avg:
            plt.plot(x, favg, marker="o", linewidth=2, label="Fitness avg")
        if plot_best:
            plt.plot(x, fbest, marker="o", linewidth=2, label="Best fitness")

        if title is None:
            run_name = self.run.get("run_name", "unknown_run")
            title = f"Fitness Evolution over Generations ({run_name})"

        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")

        self._apply_xticks(x, xtick_step=xtick_step)

        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if ylim is not None:
            plt.ylim(ylim)

        # 저장
        self._maybe_savefig(
            save_path=save_path,
            save_dir=save_dir,
            default_stem="fitness_trend",
            ext=save_ext,
            save_dpi=save_dpi,
            overwrite=overwrite,
        )

        if show:
            plt.show()

        if close:
            plt.close()

        return None  # 기존 반환 유지
