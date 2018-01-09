import numpy as np

class EventMemory:

    def __init__(self, n):
        self.n = n
        self.events = []

    def record_events(self, events):
        self.events.append(events)

    def novelty_reward(self, events):
        if len(self.events) == 0:
            return 0
        norm = self.events / np.linalg.norm(self.events)
        neg_norm = np.subtract(np.ones(self.n), norm)
        r = np.multiply(neg_norm, events)
        return np.sum(r)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

'''
event_memory = EventMemory(4)
event_memory.record_events([0.1,1,1,10])
print(event_memory.novelty_reward([1,0,0,0]))
print(event_memory.novelty_reward([0,1,0,0]))
print(event_memory.novelty_reward([0,0,0,1]))
'''