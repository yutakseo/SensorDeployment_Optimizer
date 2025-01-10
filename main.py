import os, sys, time, importlib, json
import importlib
import numpy as np
from cpuinfo import get_cpu_info
from Visual import *
from _HarrisCorner.cv_detector import *
from _SensorModule import Sensor

 
# 사용할 알고리즘
from _Algorithm.GA import *
from _Algorithm.PSO import *

class Main:
    def __init__(self, map_name, coverage, generation):
        # 동적으로 MAP 모듈 임포트
        map_module_path = f"__MAPS__.validation_maps.{map_name}"
        map_module = importlib.import_module(map_module_path)
        self.MAP = np.array(getattr(map_module, "MAP"))
        
        self.vis = VisualTool()
        self.coverage = coverage
        self.GEN = generation
        self.map_name = map_name

    def run(self):
        start = time.time()

        #최외곽 지점 추출 및 배치
        corner_instance =  HarrisCorner(self.MAP)
        corner_points = corner_instance.extract(corner_instance.harrisCorner(corner_instance.gaussianBlur(self.MAP)))

        #알고리즘 선택 및 실행
        #inner_points = sensor_GA(self.MAP, self.coverage, self.GEN).run()
        inner_points = SensorPlacementPSO(self.MAP, self.coverage, self.GEN).optimize()
        
        #결과 처리
        positions = corner_points + inner_points
        positions = [(y, x) for x, y in positions]
        runtime = time.time() - start
        num_sensor = len(positions)

        print(positions)
        print(f"경과시간(초) : {runtime:.4f}")
        print(f"총 센서 수 : {num_sensor}")

        # 센서 배치 형태 시각화
        self.vis.showBinaryMap_circle("RESULT", self.MAP, self.coverage, sensor_positions=positions)
        # 메타데이터 기록
        self.record_metadata(runtime, num_sensor, positions)
        
        return positions

    def record_metadata(self, runtime, num_sensor, sensor_positions, output_dir="__RESULTS__"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"result_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_info = get_cpu_info()['brand_raw']

        # numpy 타입을 파이썬 기본 타입으로 변환
        sensor_positions = [(int(pos[0]), int(pos[1])) for pos in sensor_positions]

        # 메타데이터를 JSON 형식으로 저장
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



if __name__ == "__main__":
    for i in range(1):
        map_name = "mid_uav"
        algorithm = Main(map_name, 40, 50).run()
