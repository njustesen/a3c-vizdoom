import numpy as np

class EventMemory:

    def __init__(self, n, alpha):
        self.n = n
        self.alpha = alpha
        self.events = np.ones(n)

    def record_events(self, events):
        dif = np.subtract(events, self.events)
        mul = np.multiply(self.alpha, dif)
        self.events = np.add(mul, self.events)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def novelty_reward(self, events):
        norm = self.events / np.linalg.norm(self.events)
        neg_norm = np.subtract(np.ones(self.n), norm)
        r = np.multiply(neg_norm, events)
        return np.sum(r)

'''
d = EventMemory(4, 0.05)
d.events = np.array([25,10,2,1])
events = [320,0,0,1]
for i in range(1):
    d.record_events(events)
print(d.events)

v = [1,2,3,4]
print(d.softmax(v))
print(v/(np.linalg.norm(v)))
'''