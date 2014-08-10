"""
The MIT License (MIT)

Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import random
import math
from collections import deque

class Diffusions(object):

    def __init__(self, diffusions=None):
        self.diffusions = diffusions

    def where_node_is_infected(self, node):
        for d in self.diffusions:
            if d.is_infected(node):
                yield d

    def where_node_is_never_infected(self, node):
        for d in self.diffusions:
            if d.is_never_infected(node):
                yield d

    def __getitem__(self, item):
        return self.diffusions[item]

class Diffusion(object):

    _lambda = 1.0
   
    def __init__(self, A, cascade=None):
        if cascade:
            self.times = cascade
            self.__length = len(cascade)
            self.__A = A
        else:
            self.__length = len(A)

            self.times = [-1] * self.__length

            seed = random.randint(0, self.__length - 1)

            self.to_propagate = deque([seed])
            self.susceptible = deque([i for i in xrange(self.__length) if i != seed])
            self.times[seed] = 0
            self.__A = A

            self.propagate()

    def __len__(self):
        return self.__length

    def is_infected_before(self, a, b):
        return all([self.times[a] < self.times[b],
                    self.times[a] != -1,
                    self.times[b] != -1])

    def is_infected(self, node):
        return self.times[node] > -1

    def is_never_infected(self, node):
        return self.times[node] == -1

    def tau(self, node):
        return self.times[node]

    def sample(self):
        return random.expovariate(lambd=self._lambda)

    def e(self, x):
        return self._lambda * math.exp(-x * self._lambda)

    def w(self, ti, tj):
        return self.e(ti - tj)

    def propagate(self):
        while self.to_propagate:
            infected = self.to_propagate.popleft()
            parent_time = self.times[infected]
            for i in xrange(len(self.susceptible)):
                susceptible = self.susceptible.popleft()
                prob = self.__A[infected][susceptible]
                if random.random() < prob:
                    self.times[susceptible] = parent_time + self.sample()
                    self.to_propagate.append(susceptible)
                else:
                    self.susceptible.append(susceptible)


if __name__ == '__main__':
    A = [[0.15, 0.2, 0.05, 0.2],
         [0.025, 0.25, 0.3, 0.6],
         [0.1, 0.1, 0.1, 0.1],
         [0.25, 0.45, 0.7, 0.8]]

    for i in range(10):
        d = Diffusion(A)
        print d.times
