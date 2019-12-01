import numpy as np
from PIL import Image

# Maximum depth of the Kinect's sensor, in meters
MAX_DEPTH = 10.0

def depth_rel_to_depth_abs(depth_rel):
    """Projects a depth image from internal Kinect coordinates to world coordinates.

    The absolute 3D space is defined by a horizontal plane made from the X and Z axes,
    with the Y axis pointing up.

    The final result is in meters."""
    depth_rel_array = np.asarray(depth_rel)

    DEPTH_PARAM_1 = 351.3
    DEPTH_PARAM_2 = 1092.5

    depth_abs_array = DEPTH_PARAM_1 / (DEPTH_PARAM_2 - depth_rel_array)
    #print("Max depth in image", np.max(depth_abs_array))
    depth_out_array = np.clip(depth_abs_array, 0, MAX_DEPTH)
    #print("Max depth in image after clipping", np.max(depth_out_array))

    depth_out = Image.fromarray(np.uint8(depth_out_array))

    return depth_out
