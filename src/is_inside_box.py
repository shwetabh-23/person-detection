def is_inside(box1, box2):

    x1_inside = box2[0] <= box1[0] <= box1[2] <= box2[2]
    y1_inside = box2[1] <= box1[1] <= box1[3] <= box2[3]

    return x1_inside and y1_inside

