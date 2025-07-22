from math import sqrt
import math

def DBSE(nusc,curr_token,ori_dets):
    sample_data_token = nusc.field2token('sample_data','sample_token',curr_token)[0]
    ego_pose_token = nusc.get('sample_data',sample_data_token)['ego_pose_token']
    sensor_pose = nusc.get('ego_pose',ego_pose_token)['translation']

    for obj in list(ori_dets):
            obj_pos = obj['translation']
            dist = calculate_dist_3d(sensor_pose,obj_pos)
            obj['dist'] = dist
            obj['detection_score'] =  get_DBSE_score(dist,obj)

    return ori_dets
def calculate_dist_3d(p1,p2):
    a1 = (p1[0]-p2[0]) * (p1[0]-p2[0])
    a2 = (p1[1]-p2[1]) * (p1[1]-p2[1])
    a3 = (p1[2]-p2[2]) * (p1[2]-p2[2])
    dist = sqrt(a1+a2+a3)
    return dist

def get_DBSE_score(dist,obj):
    try:
        score = obj['detection_score']
        obj_type = obj['detection_name']
    except:
        score = obj['tracking_score']
        obj_type = obj['tracking_name']

    if obj_type == 'pedestrian':
        # w = 1/(dist**0.15)+0.1
        w = 1

    elif obj_type == 'car':
        w = 1/(dist**0.01)+0.1

    elif obj_type == 'bus':
        w = math.exp(-(dist/70)**2)+0.1

    elif obj_type == 'truck':
        w = 1

    elif obj_type == 'motorcycle':
        # w = math.exp(-(dist/80)**2)+0.1
        w = 1

    elif obj_type == 'bicycle':
        w = math.exp(-(dist/90)**2)+0.2

    elif obj_type == 'trailer':
        w = 1/(dist**0.01)+0.1

    else:
        w = 1

    score = score * w

    # w = 1.5
    # if(dist > 55 and score > 0.3 and score < 0.5):
    #     score = score * w
    
    return min(1,score)

