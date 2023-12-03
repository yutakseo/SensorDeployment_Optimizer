import os, sys
import numpy as np
import time
from itertools import combinations

__file__ = os.getcwd()
__root__ = os.path.dirname(__file__)


sensor_module_path = os. path.join(__file__, "SensorModule")
sys.path.append(sensor_module_path)
from Sensor import *
from corner_placement import *



def non_cover(map:list):
    cord_list = []
    for i in range(len(map)):
        for j in range(len(map[0])):
            if (map[i][j] == 1):
                cord_list.append((j,i))
    return cord_list

def fill_sensor(map:list, cover):
    cord_list = non_cover(map)
    for i in range(len(cord_list)):
        sensor_instance = Sensor(map, cord_list[i], cover)
        sensor_instance.deploy_sensor()
    return map

def is_full(map:list):
    true = 0
    false = 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j] == 1:    
                if (map[i][j] // 10) != 0:
                    true += 1
                else:
                    false += 1
    if false > 0:
        return False
    else:
        return True

def combination_cover(map:list, cover):
    cord_list = non_cover(map)  #[(1,1), (2,2), ...]
    
    i = 1
    while i < len(cord_list):
        greedy_cord = (list(combinations(cord_list, i)))
        
        print(greedy_cord)
        
        for j in range(len(greedy_cord)):    
            for k in range(i):
                sensor_instance = Sensor(map, (greedy_cord[j][k][0],greedy_cord[j][k][1]), cover)
                sensor_instance.deploy_sensor()
        
        if is_full(map) == True:
            break
        else:
            i+=1
    return map
        
    
            #좌표리스트에서 좌표들을 제거하는 알고리즘 개발 필요!!!

def greedy_algorithm(map:list, cover):
    data = corner_sensor_map(map, cover, 0,0)
    none_coverd_area = non_cover(data)
    cord = []
    
    full_map = fill_sensor(data, cover)
    for i in range(len(none_coverd_area)):
        temp_sensor = Sensor(map, none_coverd_area[i], cover) 
        temp_sensor.withdraw_sensor()
        #show.show_jetmap("",data)
        
        if is_full(map) == False:
            cord.append(none_coverd_area[i])
            temp_sensor.deploy_sensor()
        elif is_full(map) ==True:
            pass
    return cord

def greedy_algorithm2(map_data, cover):
    # 맵 복사
    current_map = [row[:] for row in map_data]
    # 처음에는 모든 위치가 후보
    candidate_positions = non_cover(map_data)
    while candidate_positions:
        # 후보 위치 중 하나를 선택
        current_position = candidate_positions.pop()
        # 센서 배치
        sensor_instance = Sensor(current_map, current_position, cover)
        sensor_instance.deploy_sensor()
        # 만약 맵이 전부 채워졌다면 종료
        if is_full(current_map):
            break
        # 새롭게 미방문인 위치들을 후보로 업데이트
        candidate_positions = [pos for pos in non_cover(current_map) if pos not in candidate_positions]

    return current_map
