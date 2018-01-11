import numpy as np
import a3c_constants as constants

class EventMemory:

    def __init__(self, n, capacity):
        self.n = n
        self.capacity = capacity
        self.idx = 0
        self.events = []

    def record_events(self, events):
        if len(self.events) < self.capacity:
            self.events.append(events)
        else:
            self.events[self.idx] = events
            if self.idx + 1 < self.capacity:
                self.idx += 1
            else:
                self.idx = 0

    def novelty_reward(self, events):
        if len(self.events) == 0:
            return 0
        mean = np.mean(self.events, axis=0)
        np.clip(mean, constants.MEAN_EVENT_CLIP, constants.EPISODE_TIMEOUT, out=mean)
        div = np.divide(np.ones(self.n), mean)
        mul = np.multiply(div, events)
        return np.sum(mul)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

'''
event_memory = EventMemory(4,2)
event_memory.record_events([0.0,1,1,10])
event_memory.record_events([0.2,1,1,5])
print(event_memory.novelty_reward([1,0,0,0]))
print(event_memory.novelty_reward([0,1,0,0]))
print(event_memory.novelty_reward([0,0,0,1]))
'''