from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class Analyzer:
    run: Dict[str, Any]

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
    ) -> Tuple[List[int], List[float], List[float], List[int]]:
        """
        진화추이 시각화

        Parameters
        ----------
        include_corners : bool
            True면 inner + corner 센서 수 사용
        corner_count : Optional[int]
            corner 센서 수 강제 지정 (None이면 final.corner_points 기준)
        title : Optional[str]
            플롯 제목 (None이면 기본 제목 자동 생성)
        figsize : (float, float)
            matplotlib figure size (inch)
        dpi : int
            figure DPI (논문용이면 300 권장)
        show : bool
            plt.show() 호출 여부
        """
        gens = self._get_generations()

        x = [int(g["gen"]) for g in gens]
        smin = [float(g["sensors_min"]) for g in gens]
        smax = [float(g["sensors_max"]) for g in gens]

        fallback_corner_count = (
            self._default_corner_count()
            if corner_count is None
            else int(corner_count)
        )

        best_counts: List[int] = []
        for g in gens:
            inner = self._best_inner_count(g)
            if include_corners:
                corners = self._best_corner_count(g, fallback_corner_count)
                best_counts.append(inner + corners)
            else:
                best_counts.append(inner)

        # -------- Plot --------
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
            label=(
                "Best sensor count (inner + corner)"
                if include_corners
                else "Best sensor count (inner only)"
            ),
        )

        if title is None:
            run_name = self.run.get("run_name", "unknown_run")
            title = f"GA evolution trend: sensor count vs generation ({run_name})"

        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Number of sensors")
        plt.xticks(x)
        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if show:
            plt.show()

        return None


    def _extract_best_coverage(self, g: Dict[str, Any]) -> float:
        """
        세대별 best coverage 추출.
        - 1순위: best_coverage
        - 2순위: best_fitness (coverage와 동일 의미로 저장된 경우 fallback)
        """
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
    ) -> Tuple[List[int], List[float]]:
        """
        커버리지(coverage) 진화 추이 시각화

        Returns:
            (x, best_coverage)
        """
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
        plt.xticks(x)
        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if ylim is not None:
            plt.ylim(ylim)

        if show:
            plt.show()

        return None
    
    


    # Analyzer 클래스 내부에 추가하세요.
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
    ) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
        """
        적합도(fitness) 진화 추이 시각화

        - 음영: fitness_min ~ fitness_max
        - 라인: fitness_avg (옵션), best_fitness (옵션)

        Returns:
            (x, fmin, fmax, favg, fbest)
        """
        gens = self._get_generations()

        x = [int(g["gen"]) for g in gens]
        fmin = [float(g["fitness_min"]) for g in gens]
        fmax = [float(g["fitness_max"]) for g in gens]
        favg = [float(g["fitness_avg"]) for g in gens]
        fbest = [float(g.get("best_fitness", g["fitness_max"])) for g in gens]  # best_fitness 없으면 max로 대체

        plt.figure(figsize=figsize, dpi=dpi)

        # 분포(최소~최대) 음영
        plt.fill_between(
            x,
            fmin,
            fmax,
            alpha=0.25,
            label="Fitness range (min–max)",
        )

        # 평균/최고 라인
        if plot_avg:
            plt.plot(
                x,
                favg,
                marker="o",
                linewidth=2,
                label="Fitness avg",
            )
        if plot_best:
            plt.plot(
                x,
                fbest,
                marker="o",
                linewidth=2,
                label="Best fitness",
            )

        if title is None:
            run_name = self.run.get("run_name", "unknown_run")
            title = f"Fitness Evolution over Generations ({run_name})"

        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.xticks(x)
        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if ylim is not None:
            plt.ylim(ylim)

        if show:
            plt.show()

        return None