import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def generate_permutations(objects, boxes):
    permutations = []
    for box in boxes:
        for obj in objects:
            permutations.append((obj, box))
    return permutations

def check_dimension(objects, box):
    vol = 0

    box_vol = box.dimensions[0] * box.dimensions[1] * box.dimensions[2]

    for obj in objects:

        if max(obj.dimensions) > max(box.dimensions):
            return False
        
        vol += obj.dimensions[0] * obj.dimensions[1] * obj.dimensions[2]

    if vol > box_vol:
        return False
    
    return True



if __name__ == "__main__":
    print ("hello world")
