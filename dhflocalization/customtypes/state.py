import numpy as np


class StateHypothesis:
    def __init__(self, state_vector: np.ndarray, covar: np.ndarray):
        self.state_vector = state_vector
        self.covar = covar

    def mean(self):
        return self.state_vector

    def median(self):
        return self.state_vector


class ParticleState:
    def __init__(self, state_vectors=None):
        self.state_vectors = state_vectors

    def mean(self):
        # position_mean = np.average(self.state_vectors[:, :2], axis=0)
        # angles = self.state_vectors[:, 2]
        # angle_mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        # return np.append(position_mean, angle_mean)

        return np.average(self.state_vectors, axis=0)

    def median(self):
        return np.median(self.state_vectors, axis=0)

    def std(self):
        # mathematically not correct due to the neglection of
        # circular statistics
        return np.std(self.state_vectors, axis=0)

    @classmethod
    def init_from_gaussian(cls, mean, covar, particle_num, rng=None):
        rng = rng if rng is not None else np.random.default_rng()

        state_vectors = rng.multivariate_normal(mean, covar, particle_num)
        return cls(state_vectors=state_vectors)


class Track:
    def __init__(self, init_state=None):
        self.states = []

        if init_state:
            self.append(init_state)

    def append(self, state):
        self.states.append(state)

    def to_np_array(self):
        return np.array([state.mean() for state in self.states])

    def timesteps(self):
        return len(self.states)

    def __iter__(self):
        return iter(self.states)
