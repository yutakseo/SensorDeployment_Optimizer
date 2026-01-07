import os, sys, time, importlib, json, copy
from datetime import datetime
import numpy as np
from cpuinfo import get_cpu_info
from Tools.PlotTools import VisualTool
from OuterDeployment.HarrisCorner import *
from SensorModule.Sensor_v2 import Sensor
from Tools.MapLoader import MapLoader

# 사용할 알고리즘
from InnerDeployment.GeneticAlgorithm import SensorGA

vis=VisualTool()

map=MapLoader().load("map_280x250.bot")
vis.showJetMap(map,"test Map")

sensor = Sensor(map).deploy((50,50),20)
vis.showJetMap(map,"test Map")