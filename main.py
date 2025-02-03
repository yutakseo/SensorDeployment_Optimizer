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
        map_module_path = f"__MAPS__.{map_name}"
        map_module = importlib.import_module(map_module_path)
        self.MAP = np.array(getattr(map_module, "MAP"))
        self.vis = VisualTool()
        self.coverage = coverage
        self.GEN = generation
        self.map_name = map_name

    @staticmethod
    def record_metadata(runtime, num_sensor, sensor_positions, map_name="Unknown", output_dir="__RESULTS__"):
        """실험 결과 메타데이터 저장"""
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.now()
        time_str = now.strftime("%m-%d_%H-%M-%S")  # ✅ 날짜 형식으로 중복 방지
        file_name = f"result_{time_str}.json"
        output_file = os.path.join(output_dir, file_name)
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        cpu_info = get_cpu_info()['brand_raw']

        # ✅ 리스트가 아닐 경우 빈 리스트로 변환하여 예외 방지
        if not isinstance(sensor_positions, list):
            sensor_positions = []

        sensor_positions = [(int(pos[0]), int(pos[1])) for pos in sensor_positions]

        # 메타데이터 생성
        metadata = {
            "Timestamp": current_time,
            "CPU Name": cpu_info,
            "Runtime (s)": float(runtime),
            "Map Name": map_name,
            "Total Sensors": int(num_sensor),
            "Sensor Positions": sensor_positions
        }

        # JSON 파일 저장
        with open(output_file, mode='w', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)
        print(f"📌 메타데이터 저장 완료: {output_file}")

    def corner_deploy(self):
        """최외곽 지점 센서 배치"""
        layer_corner = copy.deepcopy(self.MAP)
        corner_instance = HarrisCorner(layer_corner)
        corner_points = corner_instance.extract(
            corner_instance.harrisCorner(corner_instance.gaussianBlur(layer_corner))
        )

        # ✅ corner_points가 리스트가 아닐 경우 빈 리스트로 변환
        if not isinstance(corner_points, list):
            corner_points = []

        # 센서 배치 (각 센서 위치를 10으로 표시)
        for pos in corner_points:
            layer_corner[pos[1], pos[0]] = 10
        return layer_corner, corner_points

    def inner_sensor_deploy(self, layer_corner):
        """GA 최적화 기반 내부 센서 배치"""
        layer_inner = copy.deepcopy(layer_corner)
        inner_points = SensorGA(layer_inner, self.coverage, self.GEN).run()

        # ✅ inner_points가 리스트가 아닐 경우 빈 리스트로 변환
        if not isinstance(inner_points, list):
            inner_points = []

        # 내부 센서 배치
        for pos in inner_points:
            layer_inner[pos[0], pos[1]] = 10
        return layer_inner, inner_points

    def run(self):
        """전체 실행 흐름"""
        start_time = time.time()

        # ✅ 현재 날짜 기반으로 폴더 생성 (월-시-분)
        now = datetime.now().strftime("%m-%d-%H-%M")
        experiment_dir = os.path.join("__RESULTS__", now)
        os.makedirs(experiment_dir, exist_ok=True)

        # 1. 최외곽 센서 배치
        layer_corner, corner_points = self.corner_deploy()

        # ✅ 최외곽 센서 배치 결과 저장
        self.vis.showJetMap_circle(
            "Corner Sensor Deployment", layer_corner, self.coverage, corner_points,
            save_path=os.path.join(experiment_dir, "corner_sensor_deployment")  # ✅ 폴더 내 저장
        )

        # 2. 내부 센서 최적화 배치
        layer_result, inner_points = self.inner_sensor_deploy(layer_corner)

        # ✅ corner_points와 inner_points가 리스트인지 확인하고, 아니면 빈 리스트로 변환
        if not isinstance(corner_points, list):
            corner_points = []
        if not isinstance(inner_points, list):
            inner_points = []

        total_sensors = len(corner_points) + len(inner_points)
        runtime = time.time() - start_time

        # ✅ 최종 센서 배치 결과를 해당 폴더에 저장
        all_sensor_positions = corner_points + inner_points
        self.vis.showJetMap_circle(
            "Final Sensor Deployment", layer_result, self.coverage, all_sensor_positions,
            save_path=os.path.join(experiment_dir, "final_sensor_deployment")  # ✅ 폴더 내 저장
        )

        # ✅ GA 진화 과정 결과도 폴더 내 저장
        self.save_checkpoint_folder = experiment_dir  # ✅ GA 결과 저장을 위한 폴더 경로 설정

        # ✅ 메타데이터 저장
        self.record_metadata(runtime, total_sensors, all_sensor_positions, self.map_name, output_dir=experiment_dir)


# 코드 본체
if __name__ == "__main__":
    for i in range(20):
        map_name = "250*280.bot"
        Main(map_name, 35, 500).run()
        
    for i in range(20):
        map_name = "250*280.mid"
        Main(map_name, 35, 500).run()
        
    for i in range(20):
        map_name = "250*280.top"
        Main(map_name, 35, 500).run()