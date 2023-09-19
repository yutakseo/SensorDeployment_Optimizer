from TEST_DATASET import *

def createCordinate(data):
    grid = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 1:
                grid.append("("+str(j+1)+","+str(i+1)+")")
    return grid

print(createCordinate(test_data4))