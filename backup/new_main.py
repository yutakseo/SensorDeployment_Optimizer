import numpy as np
from InnerDeployment.GeneticAlgorithm_cuda import SensorGA
# 1. 맵 데이터 불러오기 (예시: numpy 2D 배열)
from Tools.MapLoader import MapLoader  # 예시 함수
input_map = np.array(MapLoader("map_250x280.bot").MAP)


# 2. 파라미터 설정
coverage_radius = 45
num_generations = 100
results_dir = "./ga_results"

# 3. GA 인스턴스 생성
ga = SensorGA(
    input_map=input_map,
    coverage=coverage_radius,
    generations=num_generations,
    results_dir=results_dir,
    initial_population_size=1000,
    next_population_size=50,
    candidate_population_size=100
)

# 4. 실행!
result = ga.run()

# 5. 결과 출력
print("=== Best Result ===")
print(f"Fitness Score: {result['fitness_score']:.4f}")
print(f"Number of Sensors: {result['num_sensors']}")
print(f"Best Chromosome: {result['best_chromosome']}")