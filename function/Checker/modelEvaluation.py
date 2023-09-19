import math
from TEST_DATASET import *
from createArray import visual_array

#거리를 계산하는 함수
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def evaluation(answer, output):
    #정답데이터의 코너 개수와 좌표
    corner_position = []
    #출력데이터의 코너 개수와 좌표
    output_position = []
    
    #이중배열 반복문 실행
    for i in range(len(answer)):
        for j in range(len(answer[0])):
            if answer[i][j] != 0:
                corner_position.append([i,j])
            if output[i][j] != 0:
                output_position.append([i,j])
    
    #정답데이터의 코너 개수와 출력데이터의 코너 개수가 다를 경우의 평가
    #두 데이터의 개수가 다를 때
    if len(corner_position) != len(output_position):
        if len(corner_position) > len(output_position):
            return len(output_position)/len(corner_position)*100
        elif len(output_position) > len(corner_position):
            return (len(corner_position)-(len(output_position)-len(corner_position)))/len(corner_position)*100
    #두 데이터의 개수가 같을 때
    else:
        #쌍비교
        closest_pair = []
        for point1 in corner_position:
            min_distance = float('inf')
            pair = []
            for point2 in output_position:
                distance = calculate_distance(point1, point2)
                if distance <= min_distance:
                    min_distance = distance
                    pair = [point1, point2]
            closest_pair.append(pair)   #거리가 최소가 되는 쌍들을 생성
            print(closest_pair)
        
        #거리값들의 리스트 생성
        distance_list = []
        for i in closest_pair:    
            distance_list.append(calculate_distance(i[0], i[1]))
            

        #각각의 쌍들에서의 거리값을 반영한 점수 계산
        d_lenth = 0.0
        not_zero = 0
        for i in distance_list:
            if i == 0:
                pass
            else:
                d_lenth += i
                not_zero += 1
        print(d_lenth)
        return 100*(len(distance_list)-not_zero)/len(distance_list) + 1/(1+d_lenth)*(not_zero/len(distance_list))


#정답 데이터
source_ex = [[1,0,0],
             [0,0,0],
             [0,0,0]]

#출력 데이터
compare_ex = [[0,0,1],
              [0,0,0],   
              [0,0,0]]
        
print(evaluation(source_ex, compare_ex))