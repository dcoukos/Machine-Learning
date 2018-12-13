import random
import numpy as np


def set_random(my_seed=0):
    '''Ensures model reproducibility.'''
    random.seed(my_seed)
    np.random.seed(my_seed)
