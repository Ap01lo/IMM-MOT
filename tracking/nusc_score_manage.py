"""
assign tracklet confidence score 
predict, update, punish tracklet score under category-specific way.

TODO: to support Confidence-based method to init and kill tracklets
Code URL: CBMOT(https://github.com/cogsys-tuebingen/CBMOT)
"""
import pdb
from motion_module.nusc_object import FrameObject
import numpy as np
import math

class ScoreObject:
    def __init__(self) -> None:
        self.previous_score = -1
        self.predict_score = -1
        self.update_score = -1
        self.det_score = 0

        self.raw_score = self.final_score = -1
        
    def __repr__(self) -> str:
        repr_str = 'Previous score: {}, Predict score: {}, Update score: {}'
        return repr_str.format(self.previous_score, self.predict_score, self.update_score)

class ScoreManagement:
    def __init__(self, timestamp: int, cfg: dict, cls_label: int, det_infos: dict) -> None:
        self.initstamp = timestamp
        # self.frame_objects, self.dr = {}, cfg['life_cycle']['decay_rate'][cls_label]
        self.cfg = cfg
        self.frame_objects = {}
        self.dr = cfg['life_cycle']['decay_rate'][cls_label]

        self.decay_index = list(range(0,-50,-1))
        # self.f = lambda x: np.exp(self.cfg['life_cycle']['wa'] * x)
        self.f = lambda x: (((math.atan(x)) / 1.7) + 1) / 2
        self.weight = [self.f(idx) for idx in self.decay_index]
        self.wft = [1]
        self.cls_label = cls_label

        score_object = ScoreObject()
        score_object.raw_score = score_object.final_score = det_infos['nusc_box'].score
        score_object.predict_score = score_object.update_score = det_infos['nusc_box'].score
        score_object.det_score = det_infos['nusc_box'].score

        self.frame_objects[self.initstamp] = score_object
        self.min_hit, self.max_age = self.cfg['life_cycle']['min_hit'][cls_label], self.cfg['life_cycle']['max_age'][cls_label]
        self.time_since_update, self.hit = 0, 1
        self.curr_time = self.init_time = timestamp
        self.state = 'active' if self.min_hit <= 1 or timestamp <= self.min_hit else 'tentative'
        self.active_thres = self.cfg['life_cycle']['active_thres'][cls_label]
        self.tentative_thres = self.cfg['life_cycle']['tentative_thres'][cls_label]
    
    def predict(self, timestamp: int, pred_obj: FrameObject = None) -> None:

        if self.cfg['life_cycle']['algorithm'][self.cls_label] == 'CB':
            score_obj = ScoreObject()
            prev_score = self.frame_objects[timestamp - 1].final_score
            score_obj.raw_score, score_obj.predict_score = prev_score, prev_score * self.dr
            self.frame_objects[timestamp] = score_obj
            
            # assign tracklet score inplace
            pred_obj.predict_box.score = pred_obj.predict_infos[-5] = score_obj.predict_score
        
        elif self.cfg['life_cycle']['algorithm'][self.cls_label] == 'DW':

            self.curr_time = timestamp
            # self.time_since_update += 1

            self.wft.insert(0,0)

            score_object = ScoreObject()
            
            score_object.predict_score = self.calculate_score(self.wft,self.weight[0:len(self.wft)])
            score_object.det_score = self.frame_objects[timestamp-1].det_score

            self.frame_objects[timestamp] = score_object

            pred_obj.predict_box.score = pred_obj.predict_infos[-5] = score_object.det_score

    def update(self, timestamp: int, update_obj: FrameObject, raw_det: dict = None) -> None:
        score_obj = self.frame_objects[timestamp]
        if self.cfg['life_cycle']['algorithm'][self.cls_label] == 'CB': 
            if update_obj.update_box:
                score_obj.update_score = score_obj.final_score = raw_det['nusc_box'].score
                update_obj.update_box.score = update_obj.update_infos[-5] = raw_det['nusc_box'].score
            else:
                score_obj.final_score = score_obj.predict_score
            return

        

        if raw_det is not None:
            self.time_since_update = 0
            self.wft[0] = 1
            score_obj.update_score = self.calculate_score(self.wft,self.weight[0:len(self.wft)])
            score_obj.det_score = raw_det['nusc_box'].score
        else:
            self.time_since_update += 1
            score_obj.update_score = score_obj.predict_score

        if self.state == 'tentative':
            if score_obj.update_score > self.active_thres:
                self.state = 'active'
            elif score_obj.update_score < self.tentative_thres:
                self.state = 'dead'
            else: pass
        elif self.state == 'active':
            if score_obj.update_score < self.active_thres and score_obj.update_score > self.tentative_thres: 
                self.state = 'tentative'
            elif score_obj.update_score < self.tentative_thres:
                self.state = 'dead'
            else: pass
        else: raise Exception("dead trajectory cannot be updated")

        if raw_det is  None:
            return

        update_obj.update_box.score = update_obj.update_infos[-5] = raw_det['nusc_box'].score

    def __getitem__(self, item) -> ScoreObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)

    def calculate_score_2(self,scores,weights):
        assert len(scores) == len(weights)
        up = [0 for i in range(len(scores))]
        for i in range(len(scores)):
            up[i] = scores[i] * weights[i]
        up_total = sum(up)

        return up_total

    def calculate_score(self,scores,weights):
        assert len(scores) == len(weights)
        up = [0 for i in range(len(scores))]
        for i in range(len(scores)):
            up[i] = scores[i] * weights[i]
        up_total = sum(up)

        down_total = sum(weights)
        
        return up_total / down_total