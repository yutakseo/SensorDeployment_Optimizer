import pprint
import numpy as np
from __MAPS__.validation_maps.validation_site import *

# map 데이터를 가져옴
map = site().site2_ugv_expand

# numpy 배열을 1과 0으로 변환
arr = np.where(map == 1, 1, 0)

# numpy 배열을 리스트로 변환
converted_array = arr.tolist()

# pprint를 사용하여 배열을 정리된 문자열로 변환
formatted_array = pprint.pformat(converted_array, width=1000)

# 파일로 저장
with open('site2_ugv.py', 'w', encoding='utf-8') as f:
    f.write(f"converted_array = {formatted_array}\n")
