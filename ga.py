import numpy as np
import a3c_constants
import random
import sys

class Genome:
    def __init__(self, genotype, id):
        self.genotype = genotype
        self.fitness = []
        self.assigned = False
        self.id = id

class GA:
    def __init__(self, pop_size):
        self.pop = []
        self.count = 0
        for i in range(pop_size):
            genotype = np.random.rand(a3c_constants.GOAL_SIZE) * 2 - 1
            self.pop.append(Genome(genotype, self.count))
            self.count += 1

    def get_least_tested(self):
        t = 10000000000
        genome = None
        for genome in self.pop:
            if len(genome.fitness) < t and not genome.assigned and len(genome.fitness) > 0:
                t = len(genome.fitness)
                g = genome
        return genome

    def get_parents(self):
        parents = []
        arr = np.arange(len(self.pop))
        np.random.shuffle(arr)
        for i in arr:
            #if not self.pop[i].assigned:
            parents.append(self.pop[i])
            if len(parents) == 2:
                return parents
        return parents

    def crossover(self, genome, parents):
        for i in range(len(genome.genotype)):
            if random.uniform(0,1) >= 0.5:
                genome.genotype[i] = parents[0].genotype[i]
            else:
                genome.genotype[i] = parents[1].genotype[i]

    def mutate(self, genome):
        while(random.uniform(0,1) >= 0.5):
            genome.genotype[int(random.uniform(0,len(genome.genotype)))] = random.uniform(-1,1)

    def unassigned(self):
        n = 0
        for genome in self.pop:
            if not genome.assigned:
                n += 1
        return n

    def get_indiviual(self):
        if random.uniform(0,1) >= 0.5 and self.unassigned() > 2:
            genome = self.get_least_tested()
            genome.assigned = True
            return genome
        else:
            genome = Genome(genotype=np.random.rand(a3c_constants.GOAL_SIZE), id=self.count)
            self.count += 1
            genome.assigned = True
            parents = self.get_parents()
            if len(parents) < 2:
                print("Wht?")
            self.crossover(genome, parents)
            self.mutate(genome)
            return genome

    def evaluate(self, genome, fitness):
        genome.fitness.append(fitness)
        # Remove if worst inactive
        worst = 10000000000
        worst_idx = 0
        for i in range(len(self.pop)):
            if not genome.assigned:
                mean = np.mean(genome.fitness)
                if np.mean(genome.fitness) < worst:
                    worst = mean
                    worst_idx = i

        genome.assigned = False
        del self.pop[i]



'''
ga = GA(16)
for genome in ga.pop:
    print(str(genome.genotype))

print("-------")
parents = ga.get_parents()
for parent in parents:
    print(str(parents.genotype))
'''