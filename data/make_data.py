import numpy as np

from priors import PinWheelPrior, SquarePrior, SwissRollPrior


def make_3d_pin_wheel():
    prior = PinWheelPrior(2)
    samples = prior.sample(10000)
    new_samples = np.concatenate((samples, -np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)[:, None]), axis=-1)
    return new_samples

def make_3d_swiss_roll():
    prior = SwissRollPrior(2)
    samples = prior.sample(10000)
    new_samples = np.concatenate((samples, -np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)[:, None]), axis=-1)
    return new_samples


def make_3d_square():
    prior = SquarePrior(2)
    samples = prior.sample(10000)
    new_samples = np.concatenate((samples, (abs(samples[:, 0] - 1)  + abs(samples[:, 1] - 1))[:, None]), axis=-1)
    return new_samples