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
        """
        Serves as a wrapper over a list of diffusions. Makes it more idiomatic to iterate over them.

        :param list(diffusion.Diffusion) diffusions: A list of diffusions to wrap.
        """
        self.diffusions = diffusions

    def where_node_is_infected(self, node):
        """
        Returns a generator over the diffusions where `node` was infected.

        :param int node: The node for which we want to iterate over the diffusions where it was infected.

        :returns: A generator over the correct set of diffusions
        :rtype: generator
        """
        for d in self.diffusions:
            if d.is_infected(node):
                yield d

    def where_node_is_never_infected(self, node):
        """
        Returns a generator over the diffusions where `node` was never infected.

        :param int node: The node for which we want to iterate over the diffusions where it was never infected.

        :returns: A generator over the correct set of diffusions
        :rtype: generator
        """
        for d in self.diffusions:
            if d.is_never_infected(node):
                yield d

    def __getitem__(self, item):
        return self.diffusions[item]

class Diffusion(object):

    _lambda = 1.0
   
    def __init__(self, A, cascade=None):
        """
        Represents a cascade through the latent network represented by A. You can either pass A or both A and cascade.

        If you don't pass a cascade; one will be generated.

        :param list(list)|numpy.array A: An nxn adjacency matrix from which to generate a diffusion.
        :param list(float) cascade: A list of infection times where the indicies are the node names.

        """
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
        """
        Determines whether a was infected before b in this diffusion. If 
        either a or b was never infected; returns False.

        :param int a: The node infected first (when true)
        :param int b: The node infected later than a

        :returns: True if a was, in fact, infected before b
        :rtype: bool
        """
        return all([self.times[a] < self.times[b],
                    self.times[a] != -1,
                    self.times[b] != -1])

    def is_infected(self, node):
        """
        Determines if `node` is infected in this diffusion.

        :param int node: The node in which we're interested

        :returns: True if `node` was infected in this diffusion. False otherwise.
        :rtype: bool
        """
        return self.times[node] > -1

    def is_never_infected(self, node):
        """
        Determines if `node` is never infected in this diffusion.

        :param int node: The node in which we're interested.

        :returns: True if `node` was never infected in this diffusion. False otherwise.
        :rtype: bool
        """
        return self.times[node] == -1

    def sample(self):
        return random.expovariate(lambd=self._lambda)

    def e(self, x):
        """
        Returns the value of the exponential distribution at x.

        lambda * e^(-x*lambda)

        :param float x: The domain value to return the 
            associated probability in the exponential distribution.

        :returns: The associated probability for x
        :rtype: float
        """
        return self._lambda * math.exp(-x * self._lambda)

    def w(self, ti, tj):
        """
        Returns the probability of node i having been infected by node j based on timing.

        :param float ti: The time node i was infected.
        :param float tj: The time node j was infected.

        :returns: The probability in the exponential distribution associated with x=(ti - tj)
        :rtype: float
        """
        return self.e(ti - tj)

    def propagate(self):
        """
        Propagates infections through the graph with probabilities of Aji and w(t) 
        """
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
