import os
__file__ = os.getcwd()
__root__ = os.path.dirname(__file__)

import numpy as np
import pandas as pd
from copy import deepcopy


class SiteManagement():
    def load(self, fpath):
        return pd.read_excel(fpath, header=None, dtype=int).values

    def pad(self, site):
        return np.pad(site, ((1,1), (1,1)))

    def print(self, site):
        for row in site:
            print(' '.join([str(s) for s in row]))

    def plot(self, site):
        '''
        >>>>>>>>>>
        여기에 그리를 plotting하는 코드를 넣으면 되겠다.
        <<<<<<<<<<
        '''


class GridCornerDetection():
    def __init__(self, __root__):
        self.fname_grid_pattern = 'model/grid_pattern/after.txt'
        self.fpath_prid_pattern = os.path.join(__root__, self.fname_grid_pattern)

    def import_corner_type_labeled(self):
        with open(self.fpath_prid_pattern, 'r', encoding='utf-8') as f:
            labeled_types = [_type for _type in f.read().strip().split('SEP') if _type]
            
        filters = {}
        for row in labeled_types:
            corner_type, label = row.strip().split('--')
            corner_type_line = corner_type.strip().replace('\n', ' ')
            
            filters[corner_type_line] = label
            
        return filters

    def identify_surrounding(self, site, row_idx, col_idx):
        left_up = site[row_idx-1, col_idx-1]
        up = site[row_idx-1, col_idx]
        right_up = site[row_idx-1, col_idx+1]
        left = site[row_idx, col_idx-1]
        right = site[row_idx, col_idx+1]
        left_down = site[row_idx+1, col_idx-1]
        down = site[row_idx+1, col_idx]
        right_down = site[row_idx+1, col_idx+1]
        
        return [left_up, up, right_up, left, right, left_down, down, right_down]


    def detect(self, site):
        filters = self.import_corner_type_labeled()

        row_num, col_num = site.shape
        site_padded = SiteManagement().pad(site)

        vertex_map = deepcopy(site)
        for row_idx in range(row_num):
            for col_idx in range(col_num):
                row_idx_padded = row_idx + 1
                col_idx_padded = col_idx + 1
                me = site_padded[row_idx_padded, col_idx_padded]
                
                if me == 0:
                    continue
                else:
                    left_up, up, right_up, left, right, left_down, down, right_down = self.identify_surrounding(site_padded, row_idx_padded, col_idx_padded)
                    my_type = [left_up, up, right_up, left, me, right, left_down, down, right_down]
                    my_type_line = ' '.join([str(t) for t in my_type])

                    is_vertex = filters[my_type_line]
                    if is_vertex == '1':
                        vertex_map[row_idx, col_idx] = 2
                    else:
                        continue

        return vertex_map


class HarrisCornerDetection():
    def detect(self, data, parameters):
        '''
        Arguments
        ---------
        data : array
            | Input data
        parameters : dict
            | Hyperparameters used in Harris Corner Detection
        '''

        '''
        >>>>>>>>>>

        여기에 Harris Corner Detection의 코드를 넣으면 되겠다.
        대략적으로 틀만 잡은 것이니 필요에 따라 적절히 수정하렴.

        <<<<<<<<<<
        '''

        return result, duration