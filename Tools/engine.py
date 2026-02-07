# /workspace/Tools/engine.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass

from cpuinfo import get_cpu_info

from Tools.MapLoader import MapLoader
from Tools.Mask import layer_map
from Tools.Logger import GAJsonLogger

from OuterDeployment.HarrisCorner import HarrisCorner
from InnerDeployment.GeneticAlgorithm.main import SensorGA


class Experiment:
    def __init__(
        self,
        map_name: str,
        *,
        ga_init: Any = None,
        ga_run: Any = None,
        corner_cfg: Any = None,
        results_dir: str = "__RESULTS__",
        # logger options (Tools/Logger.py에서 지원하면 사용)
        logger_point_format: str = "tuple_str",
        logger_sort_points: bool = False,
    ):
        self.map_name = map_name
        self.results_dir = results_dir

        # ✅ 단일 소스 of truth (외부에서 dataclass를 주입)
        self.ga_init = ga_init
        self.ga_run = ga_run
        self.corner_cfg = corner_cfg

        self.logger_point_format = logger_point_format
        self.logger_sort_points = bool(logger_sort_points)

        # map & layers
        self.map = MapLoader().load(map_name)
        self.installable_layer = layer_map(self.map, keep_values=[2])
        self.road_layer = layer_map(self.map, keep_values=[3])
        self.jobsite_layer = layer_map(self.map, keep_values=[2, 3])

        self.corner_layer = HarrisCorner(self.jobsite_layer)

        # 입력 검증 (None이면 바로 명확히 터뜨리기)
        self._validate_configs()

    def _validate_configs(self) -> None:
        missing = []
        if self.ga_init is None:
            missing.append("ga_init")
        if self.ga_run is None:
            missing.append("ga_run")
        if self.corner_cfg is None:
            missing.append("corner_cfg")
        if missing:
            raise ValueError(
                f"Experiment configs missing: {missing}. "
                f"Pass dataclass instances (GAInitConfig/GARunConfig/CornerConfig) from experiment.py."
            )

        # dataclass가 아니어도 동작은 가능하지만 meta(asdict)가 깨질 수 있음
        # meta 기록을 위해 dataclass 권장
        for name, cfg in [("ga_init", self.ga_init), ("ga_run", self.ga_run), ("corner_cfg", self.corner_cfg)]:
            if not is_dataclass(cfg):
                raise TypeError(
                    f"{name} must be a dataclass instance (got {type(cfg)}). "
                    f"Define configs in experiment.py using @dataclass and pass them in."
                )

    def _build_meta(self, corner_candidate_len: int) -> Dict[str, Any]:
        # get_cpu_info()는 비용이 있으니 1회만 호출
        cpu = None
        try:
            info = get_cpu_info()
            cpu = info.get("brand_raw") if info else None
        except Exception:
            cpu = None

        return {
            "map_name": self.map_name,
            "created_at": datetime.now().isoformat(),
            "system": {"cpu": cpu},
            "ga_init": asdict(self.ga_init),
            "ga_run": asdict(self.ga_run),
            "corner": {**asdict(self.corner_cfg), "n_corner_candidates": int(corner_candidate_len)},
        }

    def run(self):
        # 1) corner 후보 생성 (corner_cfg 참조)
        cc = self.corner_cfg
        corner_candidate = self.corner_layer.run(
            grid=self.jobsite_layer,
            installable_layer=self.installable_layer,
            blockSize=cc.blockSize,
            ksize=cc.ksize,
            k=cc.k,
            dilate_size=cc.dilate_size,
            min_dist=cc.min_dist,
        )

        # 2) logger 생성 (meta는 cfg로부터 자동 생성)
        logger = GAJsonLogger(
            map_name=self.map_name,
            base_dir=self.results_dir,
            meta=self._build_meta(len(corner_candidate)),
            point_format=self.logger_point_format,
            sort_points=self.logger_sort_points,
        )

        # 3) SensorGA 생성 (ga_init 참조)
        gi = self.ga_init
        ga = SensorGA(
            installable_map=self.installable_layer,
            jobsite_map=self.jobsite_layer,
            coverage=gi.coverage,
            generations=gi.generations,
            corner_positions=corner_candidate,
            initial_size=gi.initial_size,
            selection_size=gi.selection_size,
            child_chromo_size=gi.child_chromo_size,
            min_sensors=gi.min_sensors,
            max_sensors=gi.max_sensors,
            init_min_sensors=getattr(gi, "init_min_sensors", None),
            init_max_sensors=getattr(gi, "init_max_sensors", None),
        )

        # 4) GA 실행 (ga_run 참조) + logger 전달
        gr = self.ga_run
        optimized_result = ga.run(
            selection_method=gr.selection_method,
            tournament_size=gr.tournament_size,
            mutation_rate=gr.mutation_rate,
            verbose=gr.verbose,
            profile=gr.profile,
            profile_every=gr.profile_every,
            profile_fitness_breakdown=gr.profile_fitness_breakdown,
            early_stop=gr.early_stop,
            early_stop_coverage=gr.early_stop_coverage,
            early_stop_patience=gr.early_stop_patience,
            return_best_only=gr.return_best_only,
            ordering_top_k=getattr(gr, "ordering_top_k", 0),
            mutation_kwargs=getattr(gr, "mutation_kwargs", None),
            parallel_workers=getattr(gr, "parallel_workers", 0),
            logger=logger,
        )

        # 5) 최종 결과 정리(기존 반환 형태 유지)
        final_points = list(optimized_result) + list(corner_candidate)

        # 6) JSON 저장
        out_path = logger.finalize(
            best_solution=getattr(ga, "best_solution", optimized_result),
            corner_points=getattr(ga, "corner_points", corner_candidate),
            fitness=float(getattr(ga, "best_fitness", float("nan"))),
            coverage=float(getattr(ga, "best_coverage", float("nan"))),
            extra={
                "final": {
                    "n_final_points": int(len(final_points)),
                    "n_inner": int(len(optimized_result)),
                    "n_corner": int(len(corner_candidate)),
                }
            },
        )

        return final_points, out_path

