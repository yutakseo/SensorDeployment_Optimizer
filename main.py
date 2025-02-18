import os, sys, time, importlib, json, copy
from datetime import datetime
import numpy as np
from cpuinfo import get_cpu_info
from _VisualModule_ import VisualTool
from _HarrisCorner.cv_detector import *
from _SensorModule import Sensor
from _SensorModule.coverage import *

# 사용할 알고리즘
from _Algorithm.GeneticAlgorithm import *


class SensorDeployment:
    def __init__(self, map_name, coverage, generation):
        self.visual_module = VisualTool()
        self.map_name = map_name
        self.coverage = coverage
        self.GEN = generation
        map_module_path = f"__MAPS__.{map_name}"
        map_module = importlib.import_module(map_module_path)
        self.MAP = np.array(getattr(map_module, "MAP"))

    @staticmethod
    def record_metadata(runtime, num_sensor, sensor_positions, map_name="Unknown", output_dir="__RESULTS__"):
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.now()
        time_str = now.strftime("%m-%d_%H-%M-%S")
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file_name = f"result_{time_str}.json"
        output_file = os.path.join(output_dir, file_name)
        cpu_info = get_cpu_info()['brand_raw']
        
        if not isinstance(sensor_positions, list):
            sensor_positions = []
        sensor_positions = [(int(pos[0]), int(pos[1])) for pos in sensor_positions]
        
        metadata = {
            "Timestamp": current_time,
            "CPU Name": cpu_info,
            "Runtime (s)": float(runtime),
            "Map Name": map_name,
            "Total Sensors": int(num_sensor),
            "Sensor Positions": sensor_positions
        }
        with open(output_file, mode='w', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)
        print(f"Result save at : {output_file}")


    #최외곽 지점 센서 배치 메서드
    def corner_deploy(self, map):
        layer_corner = copy.deepcopy(map)
        corner_instance = HarrisCorner(layer_corner)
        corner_points = corner_instance.extract(
            corner_instance.harrisCorner(corner_instance.gaussianBlur(layer_corner))
        )
        if not isinstance(corner_points, list):
            corner_points = []
        for pos in corner_points:
            layer_corner[pos[1], pos[0]] = 10
        return layer_corner, corner_points


    #내부 지점 센서 배치 메서드
    def inner_sensor_deploy(self, map, experiment_dir):
        layer_inner = copy.deepcopy(map)
        inner_layer, inner_points = SensorGA(layer_inner, self.coverage, self.GEN, results_dir=experiment_dir).run()
        if not isinstance(inner_points, list):
            inner_points = []
        for pos in inner_points:
            layer_inner[pos[1], pos[0]] = 10
        return layer_inner, inner_points


    #인스턴스 동작 메서드
    def run(self):
        start_time = time.time()
        now = datetime.now().strftime("%m-%d-%H-%M-%S")
        experiment_dir = os.path.join("__RESULTS__", now)
        os.makedirs(experiment_dir, exist_ok=True)

        #1. 최외곽 센서 배치
        layer_corner, corner_points = self.corner_deploy(self.MAP)
        self.visual_module.showJetMap_circle(
            "Corner Sensor Deployment", layer_corner, self.coverage, corner_points,
            save_path=os.path.join(experiment_dir, "corner_sensor_result")
        )

        #2. 내부 센서 최적화 배치
        layer_result, inner_points = self.inner_sensor_deploy(layer_corner, experiment_dir)
        if not isinstance(corner_points, list):
            corner_points = []
        if not isinstance(inner_points, list):
            inner_points = []

        #3. 최종 센서 배치 결과
        total_sensors = len(corner_points) + len(inner_points)
        runtime = time.time() - start_time
        all_sensor_positions = corner_points + inner_points
        self.visual_module.showJetMap_circle(
            "Final Sensor Deployment", layer_result, self.coverage, all_sensor_positions,
            save_path=os.path.join(experiment_dir, "Final_sensor_result")
        )
        self.save_checkpoint_folder = experiment_dir
        self.record_metadata(runtime, total_sensors, all_sensor_positions, self.map_name, output_dir=experiment_dir)
        
        
        #4. 수동배치 시 사용
        """all_sensor_positions = [[5,26],[17,32],[17,24],[33,30],[27,35],[41,45],[44,40],[42,33],[49,33],[46,38],
                                [56,41],[62,33],[62,19],[52,14],[57,8],[18,3]]
        self.visual_module.showJetMap_circle(
            "Final Sensor Deployment", self.MAP, self.coverage, all_sensor_positions,
            save_path=os.path.join(experiment_dir, "Manual_sensor_result")
        )         """               
                                
                                
        
                             




# 코드 본체
if __name__ == "__main__":
    """
    for i in range(1):
        map_name = "250x280.bot"
        SensorDeployment:(map_name, 20, 1).run()
    """
    
    """
    for i in range(1):
        map_name = "250x280.mid"
        SensorDeployment:(map_name, 20, 1).run()
    """
    
    for i in range(1):
        map_name = "250x280.bot"
        instance = SensorDeployment(map_name, 20, 1)
        instance.visual_module.showJetMap("Original Map", instance.MAP, filename="original_map")
        instance.run()