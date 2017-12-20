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

'''
event_memory = EventMemory(4)
event_memory.record_events([1,2,3,45])
event_memory.record_events([2,2,3,44])
event_memory.record_events([3,2,3,43])
event_memory.record_events([4,2,3,42])
event_memory.record_events([5,2,3,41])

for i in range(4):
    mean = np.mean(event_memory.events[-5:], axis=0)
    print(mean)
'''