"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""
import pdb
import sys
from math import log, exp, sqrt

import numpy as np
from filterpy.stats import logpdf
from numpy import dot, asarray, zeros, outer

from utils.math import warp_to_pi
from .nusc_object import FrameObject
from .motion_model import CV, CTRV, BICYCLE, CA, CTRA
from pre_processing import arraydet2box, concat_box_attr
from pyquaternion import Quaternion
from filterpy.common import kinematic_kf
from filterpy.kalman import IMMEstimator


class KalmanFilter:
    """kalman filter interface
    """

    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos['seq_id']
        self.initstamp = self.timestamp = timestamp
        self.tracking_id, self.class_label = track_id, det_infos['np_array'][-1]
        self.model = config['motion_model']['model'][self.class_label]
        self.dt, self.has_velo = config['basic']['LiDAR_interval'], config['basic']['has_velo']
        # init FrameObject for each frame
        self.state, self.frame_objects = None, {}

    def initialize(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        pass

    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """
        pass

    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass

    def getMeasureInfo(self, det: dict = None) -> np.array:
        """convert det box to the measurement info for updating filter
        [x, y, z, w, h, l, (vx, vy, optional), ry]
        Args:
            det (dict, optional): same as self.init. Defaults to None.

        Returns:
            np.array: measurement for updating filter
        """
        if det is None: raise "detection cannot be None"

        mea_attr = ('center', 'wlh', 'velocity', 'yaw') if self.has_velo else ('center', 'wlh', 'yaw')
        list_det = concat_box_attr(det['nusc_box'], *mea_attr)
        if self.has_velo: list_det.pop(8)

        # only for debug, delete on the release version
        # ensure the measure yaw goes around [0, 0, 1]
        assert list_det[-1] == det['nusc_box'].orientation.radians and det['nusc_box'].orientation.axis[-1] >= 0
        assert len(list_det) == 9 if self.has_velo else 7

        return np.mat(list_det).T

    def addFrameObject(self, timestamp: int, tra_info: dict, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data 
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (dict): Trajectory state estimated by Kalman filter, 
            {
                'exter_state': np.array, for output file. 
                               [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
                'inner_state': np.array, for state estimation. 
                               varies by motion model
            }
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        # corner case, no tra info
        if mode is None: return

        # data format conversion
        inner_info, exter_info = tra_info['inner_state'], tra_info['exter_state']
        extra_info = np.array([self.tracking_id, self.seq_id, timestamp])
        box_info, bm_info = arraydet2box(exter_info, np.array([self.tracking_id]))

        # update each frame infos 
        if mode == 'update':
            frame_object = self.frame_objects[timestamp]
            frame_object.update_bms, frame_object.update_box = bm_info[0], box_info[0]
            frame_object.update_state, frame_object.update_infos = inner_info, np.append(exter_info, extra_info)
        elif mode == 'predict':
            frame_object = FrameObject()
            frame_object.predict_bms, frame_object.predict_box = bm_info[0], box_info[0]
            frame_object.predict_state, frame_object.predict_infos = inner_info, np.append(exter_info, extra_info)
            self.frame_objects[timestamp] = frame_object
        else:
            raise Exception('mode must be update or predict')

    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """
        
        # return state vector except tra score and tra class
        inner_state = self.model.getOutputInfo(state)
        return np.append(inner_state, np.array([-1, self.class_label]))

    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)


class LinearKalmanFilter(KalmanFilter):
    """Linear Kalman Filter for linear motion model, such as CV and CA
    """

    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict, model_imm = 0) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)

        if model_imm == 0:
            self.model = globals()[self.model](self.has_velo, self.dt) if self.model in ['CV', 'CA'] \
            else globals()['CA'](self.has_velo, self.dt)
        elif model_imm == 1:
            self.model = globals()['CV'](self.has_velo, self.dt)
        elif model_imm == 2:
            self.model = globals()['CA'](self.has_velo, self.dt)
        # set motion model, default Constant Acceleration(CA) for LKF
        # Transition and Observation Matrices are fixed in the LKF
        # Only computed only if requested via property
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:
        self.F = self.model.getTransitionF()
        self.Q = self.model.getProcessNoiseQ()
        self.SD = self.model.getStateDim()
        self.MD = self.model.getMeasureDim()
        self.P = self.model.getInitCovP(self.class_label)
        
        # state to measurement transition
        self.R = self.model.getMeaNoiseR()
        self.H = self.model.getMeaStateH()

        self._res = zeros((self.MD, 1))
        self.S = np.zeros((self.MD, self.MD))  # system uncertainty

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array']
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')

    def predict(self, timestamp: int) -> None:
        # predict state and errorcov
        self.state = self.F * self.state
        self.P = self.F * self.P * self.F.T + self.Q

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')

    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return

        # update state and errorcov
        meas_info = self.getMeasureInfo(det)
        self._res = meas_info - self.H * self.state
        self.model.warpResYawToPi(self._res)
        self._S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * self._S.I
        
        self.state += _KF_GAIN * self._res
        self.P = (np.mat(np.identity(self.SD)) - _KF_GAIN * self.H) * self.P
        
        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        self._log_likelihood = logpdf(x=self._res, cov=self._S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        self._likelihood = exp(self.log_likelihood)
        if self._likelihood == 0:
            self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        """
        self._mahalanobis = sqrt(float(dot(dot(self._res.T, self._S.I), self._res)))
        return self._mahalanobis


class ExtendKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict, model_imm = 0) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        if model_imm == 0:
            self.model = globals()[self.model](self.has_velo, self.dt) if self.model in ['CTRA', 'CTRV'] \
                        else globals()['CTRA'](self.has_velo, self.dt)
        elif model_imm == 1:
            self.model = globals()['CTRV'](self.has_velo, self.dt)
        elif model_imm == 2:
            self.model = globals()['CTRA'](self.has_velo, self.dt)

        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # Transition and Observation Matrices are changing in the EKF
        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:
        # init errorcov categoty-specific
        self.SD = self.model.getStateDim()
        self.MD = self.model.getMeasureDim()
        self.P = self.model.getInitCovP(self.class_label)

        # set noise matrix(fixed)
        self.Q = self.model.getProcessNoiseQ()
        self.R = self.model.getMeaNoiseR()

        self._res = zeros((self.MD, 1))
        self.S = np.zeros((self.MD, self.MD))  # system uncertainty

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
  
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array']
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')

    def predict(self, timestamp: int) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)

        # state and errorcov transition
        self.state = self.model.stateTransition(self.state)
        self.P = self.F * self.P * self.F.T + self.Q

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')

    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return

        # get measure infos for updating, and project state into meausre space
        meas_info = self.getMeasureInfo(det)
        state_info = self.model.StateToMeasure(self.state)

        # get state residual, and warp angle diff inplace
        self._res = meas_info - state_info  
        self.model.warpResYawToPi(self._res)

        # get jacobian matrix H using the predict state
        self.H = self.model.getMeaStateH(self.state)

        # obtain KF gain and update state and errorcov
        self._S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * self._S.I
        _I_KH = np.mat(np.identity(self.SD)) - _KF_GAIN * self.H

        self.state += _KF_GAIN * self._res
        self.P = _I_KH * self.P * _I_KH.T + _KF_GAIN * self.R * _KF_GAIN.T

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'update')

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        self._log_likelihood = logpdf(x=self._res, cov=self._S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        self._likelihood = exp(self.log_likelihood)
        if self._likelihood == 0:
            self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        self._mahalanobis = sqrt(float(dot(dot(self._res.T, self._S.I), self._res)))
        return self._mahalanobis




class IMMFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        super().__init__(timestamp, config, track_id, det_infos)

        self.M = None
        self.P = None
        self.N = None
        self.likelihood = None
        self.omega = None
        self.state_prior = None
        self.P_prior = None
        self.state_post = None
        self.P_post = None
        
        self.filter1 = globals()['LinearKalmanFilter'](timestamp, config, track_id, det_infos, 1)
        self.filter2 = globals()['LinearKalmanFilter'](timestamp, config, track_id, det_infos, 2)
        self.filter3 = globals()['ExtendKalmanFilter'](timestamp, config, track_id, det_infos, 1)
        self.filter4 = globals()['ExtendKalmanFilter'](timestamp, config, track_id, det_infos, 2)

        self.mu = config['motion_model']['mu'][self.class_label]
        self.M = np.array(config['motion_model']['M'][self.class_label])
        self.filters = [self.filter1, self.filter2, self.filter3, self.filter4]

        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:

        self.state = zeros((14, 1))
        state_shape = self.state.shape
        self.P = zeros((14, 14))
        self.N = len(self.filters)  # number of filters
        self.likelihood = zeros(self.N)
        self.omega = zeros((self.N, self.N))
        self._compute_mixing_probabilities()

        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.state_prior = self.state.copy()
        self.P_prior = self.P.copy()
        self.state_post = self.state.copy()
        self.P_post = self.P.copy()

        # get different data format tracklet's state
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array']
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')

    def _compute_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        """
        self.cbar = dot(self.mu, self.M)            
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j] * self.mu[i]) / self.cbar[j]

    def _compute_state_estimate(self):
        """
        Computes the IMM's mixed state estimate from each filter using
        the the mode probability self.mu to weight the estimates.
        """
        self.state = compute_weighted_state(self.filters, self.mu)

        self.P = compute_weighted_covariance(self.filters, self.mu, self.state)

    def predict(self, timestamp: int) -> None:

        # compute mixed initial conditions
        states, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            state = zeros(self.state.shape)
            state = compute_weighted_state(self.filters, w)
            states.append(state)

            P = zeros(self.P.shape)
            P = compute_weighted_covariance(self.filters, w, state)   
            Ps.append(P)

        #  compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
                if isinstance(f.model, CA):
                    x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega = states[i].T.tolist()[0]
                    new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta]).T
                    indices_to_delete = [13]
                    P_10x14 = np.delete(Ps[i], indices_to_delete, axis=1)
                    new_P = np.delete(P_10x14, indices_to_delete, axis=0)
                    
                elif isinstance(f.model, CV):
                    x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega = states[i].T.tolist()[0]
                    new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, theta]).T
                    indices_to_delete = [9,10,11,13]
                    P_10x14 = np.delete(Ps[i], indices_to_delete, axis=1)
                    new_P = np.delete(P_10x14, indices_to_delete, axis=0)

                elif isinstance(f.model, CTRA):
                    x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega = states[i].T.tolist()[0]
                    v = np.hypot(vx, vy)
                    a = np.hypot(ax, ay)
                    new_state = np.mat([x, y, z, w, l, h, v, a, theta, omega]).T
                    indices_to_delete = [7,8,10,11]
                    P_10x14 = np.delete(Ps[i], indices_to_delete, axis=1)
                    new_P = np.delete(P_10x14, indices_to_delete, axis=0)

                elif isinstance(f.model, CTRV):
                    x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega = states[i].T.tolist()[0]
                    v = np.hypot(vx, vy)
                    new_state = np.mat([x, y, z, w, l, h, v, theta, omega]).T
                    indices_to_delete = [7,8,9,10,11]
                    P_10x14 = np.delete(Ps[i], indices_to_delete, axis=1)
                    new_P = np.delete(P_10x14, indices_to_delete, axis=0)

                f.state = new_state.copy()
                f.P = new_P.copy()
                f.predict(timestamp)

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.state_prior = self.state.copy()
        self.P_prior = self.P.copy()

        # convert the state in filter to the output format
        self.warpStateYawToPi(self.state)  

        self.state = np.matrix(self.state)

        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')

    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return

        # get measure infos for updating, and project state into meausre space
        meas_info = self.getMeasureInfo(det)
        state_info = self.StateToMeasure(self.state)

        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.warpResYawToPi(_res)

        # Update each filter and save the updated likelihood
        for i, f in enumerate(self.filters):
            f.update(timestamp, det)
            self.likelihood[i] = f.likelihood

        # Update model probability
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self._compute_mixing_probabilities()

        # Compute the mixed IMM state and covariance and save the posterior estimate
        self._compute_state_estimate()
        self.state_post = self.state.copy()
        self.P_post = self.P.copy()

        # output updated state to the result file
        self.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info
        }
        self.addFrameObject(timestamp, tra_infos, 'update')

    def StateToMeasure(self, state: np.mat) -> np.mat:
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, a, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame

        Returns:
            np.mat: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (14, 1), "state vector number in CTRA must equal to 10"
        # x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega = state.T.tolist()[0]
        v = np.hypot(vx, vy)
        if self.has_velo:
            state_info = [x, y, z,
                          w, l, h,
                          v * np.cos(theta),
                          v * np.sin(theta),
                          theta]

        else:
            state_info = [x, y, z,
                          w, l, h,
                          theta]

        return np.mat(state_info).T

    @staticmethod
    def warpResYawToPi(res: np.mat) -> np.mat:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state: np.mat) -> np.mat:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry, ry_rate]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state
    
    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-2, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        inner_state = np.array(list_state)
        # return state vector except tra score and tra class
        return np.append(inner_state, np.array([-1, self.class_label]))
    

def expand_matrix(P, new_shape, columns_to_expand, diagonal_value=1000):
        """
        Expand matrix P to new_shape, setting diagonal values at specified columns/rows.

        :param P: Original matrix
        :param new_shape: Target shape (rows, columns)
        :param columns_to_expand: Indices of columns/rows to expand
        :param diagonal_value: Diagonal value (default: 1000)
        :return: Expanded matrix
        """
        # Create new zero matrix
        new_P = np.zeros(new_shape)
    
        # Calculate position of original matrix in new matrix
        rows_to_keep = [i for i in range(new_shape[0]) if i not in columns_to_expand]
        cols_to_keep = [j for j in range(new_shape[1]) if j not in columns_to_expand]
    
        # Copy original matrix to new matrix
        new_P[np.ix_(rows_to_keep, cols_to_keep)] = P
    
        # Set diagonal values
        for col in columns_to_expand:
            if col < new_shape[0]:
                new_P[col, col] = diagonal_value
    
        return new_P


def compute_weighted_state(filters, mu):
    """
    Compute weighted state vector by fusing states from multiple motion models.
    
    Converts heterogeneous filter states into a standardized 14-dimensional representation:
    [x, y, z, width, length, height, vx, vy, vz, ax, ay, az, yaw, yaw_rate]^T
    
    Args:
        filters (list): List of filter objects with attributes:
            - model: Motion model instance (CA/CV/CTRA/CTRV)
            - state: Current state vector (numpy matrix)
        mu (list[float]): Weights for each filter (sum should equal 1.0)
    
    Returns:
        np.matrix: 14x1 fused state vector
    
    Raises:
        ValueError: Unsupported motion model detected
        
    Notes:
        Supported motion models and their conversion logic:
        - CA (13D → 14D): Adds omega=0
        - CV (10D → 14D): Adds acceleration=0 and omega=0
        - CTRA (10D → 14D): Converts polar velocity/acceleration to Cartesian
        - CTRV (9D → 14D): Converts polar velocity to Cartesian, adds acceleration=0
    """
    # Initialize zero state vector (14x1 matrix)
    total_state = np.mat(np.zeros((14, 1)))

    # Iterate through each filter and its corresponding weight
    # Convert filter state to standardized 14D representation based on motion model
    for f, weight in zip(filters, mu):
        if isinstance(f.model, CA):  # Constant Acceleration model
            x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta = f.state.T.tolist()[0]
            omega = 0  # Angular velocity not modeled in CA
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
        
        elif isinstance(f.model, CV):  # Constant Velocity model
            x, y, z, w, l, h, vx, vy, vz, theta = f.state.T.tolist()[0]
            # Set unmodeled parameters to zero
            ax = ay = az = omega = 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
        
        elif isinstance(f.model, CTRA):  # Constant Turn Rate and Acceleration model
            x, y, z, w, l, h, v, a, theta, omega = f.state.T.tolist()[0]
            # Convert polar coordinates to Cartesian
            vx, vy, vz = v * np.cos(theta), v * np.sin(theta), 0
            ax, ay, az = a * np.cos(theta), a * np.sin(theta), 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
        
        elif isinstance(f.model, CTRV):  # Constant Turn Rate and Velocity model
            x, y, z, w, l, h, v, theta, omega = f.state.T.tolist()[0]
            # Convert polar coordinates to Cartesian
            vx, vy, vz = v * np.cos(theta), v * np.sin(theta), 0
            # Acceleration not modeled in CTRV
            ax = ay = az = 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
        
        else:
            raise ValueError(f"Unsupported motion model: {type(f.model)}")
    
        # Apply weight and accumulate to total state
        total_state += new_state * weight

    return total_state
    
def compute_weighted_covariance(filters, mu, state):
    """
    Update the weighted covariance matrix by combining states and covariance matrices 
    from multiple filters with different motion models.
    """
    # Initialize temporary accumulation variables
    dim = state.shape[0]
    weighted_cov = np.mat(np.zeros((dim, dim)))

    for f, weight in zip(filters, mu):
        # Initialize variables for current filter
        new_state = None
        new_P = None
    
        # Convert filter state to standardized 14D representation based on motion model
        if isinstance(f.model, CA):  # Constant Acceleration model
            x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta = f.state.T.tolist()[0]
            omega = 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
            new_P = expand_matrix(f.P, (14, 14), [13])  # Expand for omega dimension
        
        elif isinstance(f.model, CV):  # Constant Velocity model
            x, y, z, w, l, h, vx, vy, vz, theta = f.state.T.tolist()[0]
            ax = ay = az = omega = 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
            new_P = expand_matrix(f.P, (14, 14), [9, 10, 11, 13])  # Expand for ax, ay, az, omega
        
        elif isinstance(f.model, CTRA):  # Constant Turn Rate and Acceleration model
            x, y, z, w, l, h, v, a, theta, omega = f.state.T.tolist()[0]
            vx, vy, vz = v * np.cos(theta), v * np.sin(theta), 0
            ax, ay, az = a * np.cos(theta), a * np.sin(theta), 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
            new_P = expand_matrix(f.P, (14, 14), [7, 8, 10, 11])  # Expand for vx, vy, ay, az
        
        elif isinstance(f.model, CTRV):  # Constant Turn Rate and Velocity model
            x, y, z, w, l, h, v, theta, omega = f.state.T.tolist()[0]
            vx, vy, vz = v * np.cos(theta), v * np.sin(theta), 0
            ax = ay = az = 0
            new_state = np.mat([x, y, z, w, l, h, vx, vy, vz, ax, ay, az, theta, omega]).T
            new_P = expand_matrix(f.P, (14, 14), [7, 8, 9, 10, 11])  # Expand for vx, vy, ax, ay, az
            
        else:
            raise ValueError(f"Unsupported motion model: {type(f.model)}")
        
        # Calculate deviation vector
        y = new_state - state
        
        # Accumulate weighted covariance components
        weighted_cov += weight * (outer(y, y) + new_P)
    
    return weighted_cov

