import os, sys, time, importlib, json, copy
from datetime import datetime
import numpy as np
from cpuinfo import get_cpu_info
from Tools.PlotTools import VisualTool
from OuterDeployment.HarrisCorner import *
from SensorModule import Sensor
from SensorModule.coverage import *
from Tools.MapLoader import MapLoader

# 사용할 알고리즘
from InnerDeployment.GeneticAlgorithm import SensorGA


class SensorDeployment:
    def __init__(self, map_name, coverage: int = 45, generation: int = 100):
        self.visual_module = VisualTool()
        self.map_name = map_name
        # 시각화용 반경과 평가용 반경을 분리
        self.coverage_for_eval = int(coverage)      # 실제 커버리지 평가 반경
        self.radius_for_plot  = coverage / 5        # 시각화 원 크기
        self.GEN = generation
        # MAP을 숫자형으로 강제 변환하여 imshow 타입 에러 방지
        self.MAP = np.asarray(MapLoader().MAP)
        if self.MAP.dtype.kind in ("U", "S", "O"):
            self.MAP = self.MAP.astype(np.float32)

    @staticmethod
    def record_metadata(runtime, num_sensor, coverage_score, sensor_positions,
                        map_name="Unknown", output_dir="__RESULTS__"):
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
            "Coverage Ratio": float(coverage_score),
            "Sensor Positions": sensor_positions
        }
        with open(output_file, mode='w', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)
        print(f"Result save at : {output_file}")

    # 최외곽 지점 센서 배치 메서드
    def corner_deploy(self, map_array):
        layer_corner = copy.deepcopy(map_array)
        corner_instance = HarrisCorner(layer_corner)
        points_corner = corner_instance.run(
            map=layer_corner,
            block_size=3,
            ksize=3,
            k=0.05,
            dilate_size=5
        )

        if not isinstance(points_corner, list):
            points_corner = []
        else:
            for pos in points_corner:
                # (x, y) → (row=y, col=x)
                layer_corner[pos[1], pos[0]] = 10

        return layer_corner, points_corner

    # 내부 지점 센서 배치 메서드
    def inner_sensor_deploy(self, map_array, experiment_dir):
        layer_inner = copy.deepcopy(map_array)
        inner_layer, inner_points, coverage_score = SensorGA(
            layer_inner, self.coverage_for_eval, self.GEN, results_dir=experiment_dir
        ).run()

        if not isinstance(inner_points, list):
            inner_points = []
        for pos in inner_points:
            layer_inner[pos[1], pos[0]] = 10

        return inner_layer, inner_points, coverage_score

    # 인스턴스 동작 메서드
    def run(self):
        start_time = time.time()
        now = datetime.now().strftime("%m-%d-%H-%M-%S")
        experiment_dir = os.path.join("__RESULTS__", now)
        os.makedirs(experiment_dir, exist_ok=True)

        # 1. 최외곽 센서 배치
        layer_corner, points_corner = self.corner_deploy(self.MAP)
        self.visual_module.showJetMap_circle(
            map_data=layer_corner,
            sensor_positions=points_corner,
            title="Corner Sensor Deployment",
            radius=self.radius_for_plot,     # 시각화 반경
            filename="corner_sensor_result",
            save_path=experiment_dir
        )

        # 2. 내부 센서 최적화 배치
        layer_result, inner_points, _ = self.inner_sensor_deploy(layer_corner, experiment_dir)
        if not isinstance(points_corner, list):
            points_corner = []
        if not isinstance(inner_points, list):
            inner_points = []

        # 3. 최종 센서 배치 결과 (코너 + 내부)
        all_sensor_positions = points_corner + inner_points
        total_sensors = len(all_sensor_positions)
        runtime = time.time() - start_time

        # 3.2. 커버리지 비율 계산
        sensor_model = Sensor(self.MAP)
        uncovered = sensor_model.deploy(all_sensor_positions, self.coverage_for_eval)
        uncovered_score = np.sum(uncovered == 2)
        total_site = np.sum(self.MAP == 1)
        print("전체영역(1)", int(total_site))
        print("미커버영역", int(uncovered_score))
        coverage_score = ((total_site - uncovered_score) / total_site) * 100 if total_site > 0 else 0.0
        print("Coverage Ratio", float(coverage_score))

        # 최종 결과 시각화
        self.visual_module.showJetMap_circle(
            map_data=layer_result,
            sensor_positions=all_sensor_positions,
            title="Final Sensor Deployment",
            radius=self.radius_for_plot,
            filename="final_sensor_result",
            save_path=experiment_dir
        )

        self.save_checkpoint_folder = experiment_dir
        self.record_metadata(
            runtime, total_sensors, coverage_score, all_sensor_positions,
            self.map_name, output_dir=experiment_dir
        )

        """
        # 4. 수동배치 시 사용 예시 (인자 순서 교정!)
        # all_sensor_positions = [[2,11],[21,2],[14,17],[37,12],[34,6],[16,43]]
        self.visual_module.showJetMap(
            map_data=self.MAP,
            title="Site Map",
            filename="site_map",
            save_path=experiment_dir
        )
        self.visual_module.showJetMap_circle(
            map_data=self.MAP,
            sensor_positions=all_sensor_positions,
            title="Manual Sensor Deployment",
            radius=self.radius_for_plot,
            filename="manual_sensor_result",
            save_path=experiment_dir
        )
        """

if __name__ == "__main__":
    # 100x100
    for i in range(1):
        instance = SensorDeployment(map_name="map_100x100.top", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
    for i in range(10):
        instance = SensorDeployment(map_name="map_100x100.mid", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
    for i in range(10):
        instance = SensorDeployment(map_name="map_100x100.bot", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()

    # 200x200
    for i in range(1):
        instance = SensorDeployment(map_name="map_200x200.top", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
    for i in range(10):
        instance = SensorDeployment(map_name="map_200x200.mid", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
    for i in range(10):
        instance = SensorDeployment(map_name="map_200x200.bot", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()

    # 250x280
    for i in range(1):
        instance = SensorDeployment(map_name="map_250x280.top", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
    for i in range(10):
        instance = SensorDeployment(map_name="map_250x280.mid", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
    for i in range(10):
        instance = SensorDeployment(map_name="map_250x280.bot", coverage=45, generation=100)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()

    # Large
    for i in range(1):
        instance = SensorDeployment(map_name="map_570x1100.large", coverage=45, generation=500)
        instance.visual_module.showJetMap(map_data=instance.MAP, title="Original Map", filename="original_map")
        instance.run()
