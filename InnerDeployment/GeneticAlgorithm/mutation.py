import random

def mutation(self, chromosome) -> list:
    genes, score = chromosome  # chromosome: (genes, fitness_score)
    genes = genes.copy()  # 원본 보호

    if score >= 100:
        # 높은 성능이면 센서 하나 제거
        if len(genes) >= 2:
            genes = genes[:-2]
    else:
        # 성능 낮으면 미커버 영역에서 센서 추가
        sensor_pos_list = [(genes[i], genes[i+1]) for i in range(0, len(genes), 2)]
        positions = self.fitnessFunc.extractUncovered(corner_positions=self.corner_positions, inner_positions=sensor_pos_list)
        if positions:
            x, y = random.choice(positions)
            genes.extend([int(x), int(y)])
    return genes