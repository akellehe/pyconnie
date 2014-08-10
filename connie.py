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

def convex_formulation(Ai, i=0, D=None):
    return outer_sum_over_minus_gamma_hat(D, Ai, i) - outer_sum_over_Bji_hat(D, Ai, i)

