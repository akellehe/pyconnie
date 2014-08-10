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

__all__ = ['max_likelihood_formulation', 'convex_formulation']

import math

"""
Max-likelihood formulation
"""
def prob_i_inf_at_ti(c, i, A):
    ti = c.times[i]
    total = 1.0
    for j, tj in enumerate(c.times):
        if not c.is_infected_before(j, i):
            continue
        total *= (1.0 - c.w(ti, tj)*A[j])
    return 1.0 - total

def prob_i_never_infected(c, i, A):
    total = 1.0
    for j, tj in enumerate(c.times):
        if j == i:
            continue
        if c.is_infected(j):
            total *= (1.0 - A[j])
    return total

def outer_sum_over_prob_i_infected(D, i, A):
    total = 1.0
    for c in D.where_node_is_infected(i):
        total *= prob_i_inf_at_ti(c, i, A)
    return total

def outer_sum_over_prob_i_not_infected(D, i, A):
    total = 1.0
    for c in D.where_node_is_never_infected(i):
        total *= prob_i_never_infected(c, i, A)
    return total

def max_likelihood_formulation(D, i, A):
    """
    This is the max-likelihood formulation, L(A; D). It is not
    convex and therefore difficult to solve. You should really
    use the convex formulation.

    :param diffusion.Diffusions D: The set of diffusions we've observed, from which we're trying to deduce Ai
    :param int i: The node we're solving for the incoming edges to.
    :param list|numpy.array Ai: An input column of Ai. Should be 1xN, where N is the number of unique nodes in D

    :returns: The likelihood of this particular guess for column i of the Adjacency matrix given the observed cascades, D.
    :rtype: float
    """
    return outer_sum_over_prob_i_not_infected(D, i, A) * outer_sum_over_prob_i_infected(D, i, A)

def to_minimize(A, i, D):
    try:
        return -math.log(max_likelihood_formulation(D, i, A))
    except ValueError:
        return float('Inf')


"""
Convex formulation
"""
def minus_gamma_hat(c, i, A):
    ti = c.times[i]
    total = 1.0
    for j, tj in enumerate(c.times):
        if not c.is_infected_before(j, i):
            continue
        total *= (1.0 - c.w(ti, tj)*A[j])
    if total == 1.0:
        # Means no j was infected before i.
        # 0.0 will be the same as if this was never attempted.
        return 0.0
    if total == 1.0:
        return -float("Inf")
    return -math.log(1.0 - total)

def inner_sum_over_Bji_hat(c, i, A):
    total = 0.0
    for j, tj in enumerate(c.times):
        if j == i:
            continue
        if c.is_infected(j):
            try:
                total += math.log(1.0 - A[j])
            except ValueError, e:
                total += float("Inf")
    return total

def penalty_term(A, i, rho):
    total = 0.0
    for j, Aij in enumerate(A):
        if j == i:
            continue
        try:
            total += math.exp(-math.log(1.0 - Aij))
        except ValueError, e:
            total += float("Inf")
    return rho * total

def outer_sum_over_minus_gamma_hat(D, A, i):
    total = 0.0
    for c in D.where_node_is_infected(i):
        total += minus_gamma_hat(c, i, A)
    return total

def outer_sum_over_Bji_hat(D, A, i):
    total = 0.0
    for c in D.where_node_is_never_infected(i):
        total += inner_sum_over_Bji_hat(c, i, A)
    return total

def convex_formulation(Ai, i=0, D=None, rho=0.0):
    """
    The convex formulation of the log-likelihood minimization problem.
    By passing A single column of A, Ai, we can minimize this function
    to arrive at the most likely column given an input set of cascades.

    :param list|numpy.array Ai: An input column of Ai. Should be 1xN, where N is the number of unique nodes in D
    :param int i: The node we're solving for the incoming edges to.
    :param diffusion.Diffusions D: The set of diffusions we've observed, from which we're trying to deduce Ai
    :param float rho: A penalty term for toggling between precision and recall. Solving with rho > 0 is very accurate for discovering edges. rho == 0 is much more precise for edge weights.

    :returns: The log-likelihood in the convex formulation of this particular guess for column i of the Adjacency matrix given the observed cascades, D.
    :rtype: float
    """
    return outer_sum_over_minus_gamma_hat(D, Ai, i) - outer_sum_over_Bji_hat(D, Ai, i) + penalty_term(Ai, i, rho)

