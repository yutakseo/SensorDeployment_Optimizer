import torch
import torch.nn as nn
import torch.optim as optimim
import torch.nn.functional as Func
import sys, os, numpy as np

top_py_dir = "/app/__MAPS__/200x200/top.py"
sys.path.append(top_py_dir)
map_module = importlib.import_module(map_module_path)
MAP = np.array(getattr(map_module, "MAP"))

class Reinforcement:
    def __init__(self, map, coverage:int) -> None:
        self.MAP = torch.tensor(map)
        self.COV = int(coverage/5)
            
    def agent():
        return None
    
    def neuralNetwork():
        return None
    
    def evaluator():
        return None

temp=Reinforcement(MAP, 45)
print(temp.MAP)
