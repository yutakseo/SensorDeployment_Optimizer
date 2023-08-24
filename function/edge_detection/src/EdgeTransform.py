def ribosome(input):
#행방향 변환작업
        row_converted_data = []
        input_data = input
        for i in range(len(input_data)): #세로축 반복
                row_data = []
                for j in range(len(input_data[0])): #가로축 반복
                        if input_data[i][j] == 1:  #기준셀이 1일때,
                            if (input_data[i][j-1] == 0) & (input_data[i][j+1] == 0): 
                                    row_data.append(1) #후미,선두=0 일때, 1 추가
                            elif (input_data[i][j-1] == 1) & (input_data[i][j+1] == 0):
                                    row_data.append(1) #후미=1,선두=0 일때, 1 추가
                            elif (input_data[i][j-1] == 0) & (input_data[i][j+1] == 1):
                                    row_data.append(1) #후미=0,선두=1 일때, 1 추가
                            else:
                                    row_data.append(0) #후미,선두=1 일때, 0 추가
                        else:
                            row_data.append(0)        
                row_data.append(0)
                row_converted_data.append(row_data)
        

    #열방향 변환작업
        column_converted_data = []        
        for j in range(len(input_data)):
                column_data = []
                for i in range(len(input_data[0])):
                        if input_data[j][i] == 1:
                                if (input_data[j-1][i] == 0) & (input_data[j+1][i] == 0):
                                        column_data.append(1)
                                elif (input_data[j-1][i] == 1) & (input_data[j+1][i] == 0):
                                        column_data.append(1)
                                elif (input_data[j-1][i] == 0) & (input_data[j+1][i] == 1):
                                        column_data.append(1)
                                else:
                                        column_data.append(0)
                        else:
                                column_data.append(0)       
                column_converted_data.append(column_data)  
                    
        
    
    #행변환 배열과 열변환 배열 통합
        converted_data = []
        for i in range(len(row_converted_data)):
                raw_data=[]
                for j in range(len(column_converted_data[0])):
                    if (row_converted_data[i][j] ==1) or (column_converted_data[i][j]==1):
                        raw_data.append(1)
                    else:
                        raw_data.append(0)
                converted_data.append(raw_data)
        
        return converted_data


