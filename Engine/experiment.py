from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass
from cpuinfo import get_cpu_info
from Engine.map_loader import MapLoader
from Engine.masks import layer_map
from Engine.logger import GAJsonLogger
from Engine.optimizers import make_inner_optimizer
from OuterDeployment.HarrisCorner import HarrisCorner


class Experiment:
    def __init__(
        self,
        map_name: str,
        *,
        ga_init: Any = None,
        ga_run: Any = None,
        optimizer_init: Any = None,
        optimizer_run: Any = None,
        corner_cfg: Any = None,
        results_dir: str = "__RESULTS__",
        # logger options (Engine/logger.py에서 지원하면 사용)
        logger_point_format: str = "tuple_str",
        logger_sort_points: bool = False,
    ):
        self.map_name = map_name
        self.results_dir = results_dir

        # Backward-compatible aliases: older callers pass ga_init/ga_run.
        self.optimizer_init = optimizer_init if optimizer_init is not None else ga_init
        self.optimizer_run = optimizer_run if optimizer_run is not None else ga_run
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
        if self.optimizer_init is None:
            missing.append("optimizer_init")
        if self.optimizer_run is None:
            missing.append("optimizer_run")
        if self.corner_cfg is None:
            missing.append("corner_cfg")
        if missing:
            raise ValueError(
                f"Experiment configs missing: {missing}. "
                f"Pass dataclass instances for optimizer_init, optimizer_run, and corner_cfg."
            )

        # meta 기록을 위해 dataclass 권장
        for name, cfg in [
            ("optimizer_init", self.optimizer_init),
            ("optimizer_run", self.optimizer_run),
            ("corner_cfg", self.corner_cfg),
        ]:
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
            "optimizer_init": asdict(self.optimizer_init),
            "optimizer_run": asdict(self.optimizer_run),
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

        # 3) Inner optimizer 생성
        oi = self.optimizer_init
        algorithm = str(
            getattr(oi, "algorithm", getattr(oi, "optimizer", "ga"))
        ).lower()
        optimizer = make_inner_optimizer(
            algorithm=algorithm,
            installable_map=self.installable_layer,
            jobsite_map=self.jobsite_layer,
            corner_positions=corner_candidate,
            init_cfg=self.optimizer_init,
            run_cfg=self.optimizer_run,
            logger=logger,
        )

        # 4) Inner optimizer 실행. 알고리즘별 인자는 strategy가 책임진다.
        optimized_result = optimizer.run()

        # 5) 최종 결과 정리(기존 반환 형태 유지)
        final_points = list(optimized_result) + list(corner_candidate)

        # 6) JSON 저장
        out_path = logger.finalize(
            best_solution=optimizer.best_solution or optimized_result,
            corner_points=optimizer.corner_points,
            fitness=optimizer.best_fitness,
            coverage=optimizer.best_coverage,
            extra={
                "final": {
                    "n_final_points": int(len(final_points)),
                    "n_inner": int(len(optimized_result)),
                    "n_corner": int(len(corner_candidate)),
                }
            },
        )

        return final_points, out_path
