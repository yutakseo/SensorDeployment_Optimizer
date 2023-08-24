from TEST_DATASET import *
from createArray import visual_array

def evaluation(src, compare):
    total_cell = 0
    true_cell = 0
    for i in range(len(src)):
        for j in range(len(src[0])):
            if src[i][j] == compare[i][j]:
                true_cell += 1
            total_cell += 1
    
    average = true_cell / total_cell * 100
    print("유사도 : ",round(average,2))
    return average

'''
print("유사도 : ",round(evaluation(test_data_mini, test_data_mini_compare),2))
visual_array("true",test_data_mini)
visual_array('model' ,test_data_mini_compare)
'''