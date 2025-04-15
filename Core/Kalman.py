import numpy as np
import torch

NUM_SECONDS_TILL_DEATH = 2.0


class Kalman:

    def __init__(self, dt, pos, device):
        self.dt = dt
        self.estimate_pos = [pos, 0.0]
        self.device = device

        self.blocks_remaining = 0
        self.observation_error = 1e-3 #1e-3
        self.observation = 0.0
        self.kalman_gain = np.array([0.0, 0.0])
        self.estimate = np.array([0.0, 0.0])
        self.mat_a = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.mat_h = np.array([1, 0])
        self.mat_q = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.process_error = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.state_covariance = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.norm_dist_params = {'mean': 0, 'std': 0.1}

        self.reset_remaining()

    @staticmethod
    def calculate_dt(hop_size, sampling_rate):
        return hop_size / sampling_rate

    def reset_remaining(self):
        self.blocks_remaining = int(NUM_SECONDS_TILL_DEATH / self.dt)

    def get_blocks_remaining(self):
        return self.blocks_remaining

    def iterate(self, new_measurement):

        self.make_observation(new_measurement)

        self.update_state_covariance()

        self.calculate_kalman_gain()

        self.update_process_error()

        return self.new_estimate()

    def make_observation(self, new_measurement):
        # print(f'new_measurement: {new_measurement}')
        self.observation = new_measurement + self.observation_error * np.random.normal(loc=self.norm_dist_params['mean'],
                                                                                       scale=self.norm_dist_params['std'])

    def update_state_covariance(self):
        self.state_covariance = np.matmul(self.mat_a, np.matmul(self.process_error, self.mat_a.transpose())) + self.mat_q

    def calculate_kalman_gain(self):
        self.kalman_gain = np.matmul(self.state_covariance, self.mat_h.transpose()) / \
                           np.matmul(self.mat_h, np.matmul(self.state_covariance, self.mat_h.transpose())) + \
                           self.observation_error * self.observation_error

    def update_process_error(self):
        self.process_error = self.state_covariance - np.matmul(self.kalman_gain, np.matmul(self.mat_h, self.state_covariance))

    def new_estimate(self):
        self.estimate = np.matmul(self.mat_a, self.estimate) + np.multiply(self.kalman_gain, (self.observation -
                                                     np.matmul(self.mat_h, np.matmul(self.mat_a, self.estimate))))

        return self.estimate[0]

    def get_estimate(self):
        return self.estimate[0]

