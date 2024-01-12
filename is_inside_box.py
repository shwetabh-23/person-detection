def is_inside(box1, box2):
    """
    Check if bounding box 1 is inside bounding box 2.

    Parameters:
    - box1: List [x1, y1, x2, y2] representing the coordinates of the first bounding box.
    - box2: List [x1, y1, x2, y2] representing the coordinates of the second bounding box.

    Returns:
    - True if box1 is inside box2, False otherwise.
    """
    x1_inside = box2[0] <= box1[0] <= box1[2] <= box2[2]
    y1_inside = box2[1] <= box1[1] <= box1[3] <= box2[3]

    return x1_inside and y1_inside


# # Example usage:
# box1 = [10, 20, 50, 60]
# box2 = [5, 15, 60, 70]

# if is_inside(box1, box2):
#     print("Box 1 is inside Box 2.")
# else:
#     print("Box 1 is not inside Box 2.")
