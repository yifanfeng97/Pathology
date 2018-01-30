import math
from . import meter
import numpy as np

class AverageValueMeter(meter.Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.sum += value * n
        self.var += (value * value) * n
        self.count += n

    def value(self):
        n = self.count
        if n == 0:
            mean, std = np.nan, np.nan
        elif n == 1:
            return self.sum, np.inf
        else:
            mean = self.sum / n
            std = math.sqrt( (self.var - n * mean * mean) / (n - 1.0) )
        return mean, std

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.var = 0.0

