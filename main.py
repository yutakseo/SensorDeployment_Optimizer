import os, sys, time, importlib, json, copy
import numpy as np
from cpuinfo import get_cpu_info
from _VisualModule import *
from _HarrisCorner.cv_detector import *
from _SensorModule import Sensor
from _SensorModule.coverage import *



# 사용할 알고리즘
from _Algorithm.GA import *
from _Algorithm.PSO import *

class Main:
    def __init__(self, map_name, coverage, generation):
        map_module_path = f"__MAPS__.validation_maps.{map_name}"
        map_module = importlib.import_module(map_module_path)
        self.MAP = np.array(getattr(map_module, "MAP"))
        self.vis = VisualTool()
        self.coverage = coverage
        self.GEN = generation
        self.map_name = map_name

    def record_metadata(self, runtime, num_sensor, sensor_positions, output_dir="__RESULTS__"):
        """실험 결과 메타데이터 저장"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"result_{time.strftime('%Y%m%d_%H%M%S')}.json")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_info = get_cpu_info()['brand_raw']

        sensor_positions = [(int(pos[0]), int(pos[1])) for pos in sensor_positions]  # numpy -> 기본 타입 변환

        metadata = {
            "Timestamp": current_time,
            "CPU Name": cpu_info,
            "Runtime (s)": float(runtime),
            "Map Name": self.map_name,
            "Total Sensors": int(num_sensor),
            "Sensor Positions": sensor_positions
        }
        with open(output_file, mode='w') as file:
            json.dump(metadata, file, separators=(',', ':'))

    def corner_deploy(self):
        """최외곽 지점 센서 배치"""
        layer_corner = copy.deepcopy(self.MAP)
        corner_instance = HarrisCorner(layer_corner)
        corner_points = corner_instance.extract(
            corner_instance.harrisCorner(corner_instance.gaussianBlur(layer_corner))
        )

        # 센서 배치 (각 센서 위치를 10으로 표시)
        for pos in corner_points:
            layer_corner[pos[1], pos[0]] = 10

        # 시각화
        self.vis.showJetMap_circle("Corner Deployment", layer_corner, self.coverage, corner_points)

        return layer_corner, corner_points

    def inner_sensor_deploy(self, layer_corner):
        """GA 최적화 기반 내부 센서 배치"""
        layer_inner = copy.deepcopy(layer_corner)
        inner_points = SensorGA(layer_inner, self.coverage, self.GEN).run()
        # 내부 센서 배치
        for pos in inner_points:
            layer_inner[pos[0], pos[1]] = 10
        # 시각화
        self.vis.showJetMap("Final Sensor Deployment", layer_inner)
        return layer_inner, inner_points

    def run(self):
        """전체 실행 흐름"""
        start_time = time.time()

        # 1. 최외곽 센서 배치
        layer_corner, corner_points = self.corner_deploy()

        # 2. 내부 센서 최적화 배치
        layer_result, inner_points = self.inner_sensor_deploy(layer_corner)

        
        total_sensors = len(corner_points) + len(inner_points)
        runtime = time.time() - start_time
        self.record_metadata(runtime, total_sensors, corner_points + inner_points)

# 코드 본체
if __name__ == "__main__":
    map_name = "bot_uav"
    Main(map_name, 30, 500).run()
