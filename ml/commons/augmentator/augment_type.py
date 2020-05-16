import random
from ml.pt.logger import PtLogger


class DualCompose:
    def __init__(self, transforms, prob=None):
        self.transforms = transforms
        self.prob = prob

    @PtLogger(debug=True)
    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    @PtLogger(debug=True)
    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.0
            x, mask = t(x, mask)
        return x, mask


class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.0
        self.second = second
        second.prob = 1.0
        self.prob = prob

    @PtLogger(debug=True)
    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask
