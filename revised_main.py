import os, sys, time, importlib, json, copy
import numpy as np
from cpuinfo import get_cpu_info
from _VisualModule import *
from _HarrisCorner.cv_detector import *
from _SensorModule import Sensor

 
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

    def run(self):
        start = time.time()

        layer_0 = self.MAP
        
        #최외곽 지점 추출 및 배치
        corner_instance =  HarrisCorner(self.MAP)
        corner_points = corner_instance.extract(corner_instance.harrisCorner(corner_instance.gaussianBlur(self.MAP)))
        self.vis.showJetMap_circle("", layer_0, self.coverage, corner_points)
        layer_corner = copy.deepcopy(layer_0)
        for pos in corner_points:
            layer_corner[pos[1], pos[0]] = 10
        self.vis.showJetMap("", layer_corner)
        
        #센서 배치



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


#코드본체
if __name__ == "__main__":
    for i in range(1):
        map_name = "bot_uav"
        algorithm = Main(map_name, 40, 50).run()
