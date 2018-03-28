from constants import *

def np_index_to_human_coord(coord):
    a, b = coord
    return (a - b + 7, min(a, b) + 1)

def human_coord_to_np_index(coord):
    # TODO
    raise NotImplementedError()
