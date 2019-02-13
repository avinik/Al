import numpy as np
import math

def entropysampling(prob):
    entropy = 0
    for p in prob:
        entropy  = entropy + p*math.log2(p)
    return entropy