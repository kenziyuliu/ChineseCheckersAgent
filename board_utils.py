import constants

def human_coord_to_np_index(coord):
    a, b = coord
    return (a - b + 7, min(a, b) + 1)
