import unittest

from scipy.optimize import fmin_tnc
import numpy

from diffusion import Diffusions
from diffusion import Diffusion

from connie import prob_i_never_infected
from connie import prob_i_inf_at_ti
from connie import minus_gamma_hat
from connie import inner_sum_over_Bji_hat
from connie import outer_sum_over_minus_gamma_hat
from connie import outer_sum_over_Bji_hat
from connie import convex_formulation
from connie import to_minimize

class ConnieTest(unittest.TestCase):
    def setUp(self):
        self.A = [[0.15,  0.2,  0.05, 0.2],
                  [0.025, 0.25, 0.3,  0.6],
                  [0.1,   0.1,  0.1,  0.1],
                  [0.25,  0.45, 0.7,  0.8]]
        self.A0 = [0.0,
                   0.025,
                   0.1,
                   0.25]
        self.A1 = [0.2,
                   0.0,
                   0.1,
                   0.45]
        self.A2 = [0.05,
                   0.3,
                   0.0,
                   0.7]
        self.A3 = [0.2,
                   0.6,
                   0.1,
                   0.0]
        self.D = Diffusions([Diffusion(self.A, cascade=[-1,     -1,       0,     -1]),
                             Diffusion(self.A, cascade=[0,      -1,      -1,     -1]),
                             Diffusion(self.A, cascade=[0,       3.5866, -1,     -1]),
                             Diffusion(self.A, cascade=[-1,     -1,       0.6115, 0]),
                             Diffusion(self.A, cascade=[1.7403,  0.8380,  0.3760, 0]),
                             Diffusion(self.A, cascade=[1.3690, -1,       0,      1.1501]),
                             Diffusion(self.A, cascade=[1.6617,  0.1125,  0.1946, 0])])

class TestConnie(ConnieTest):
    def test_prob_i_inf_at_ti(self):
        self.assertEquals(prob_i_inf_at_ti(self.D[0], 0, self.A0), 0)
        self.assertEquals(prob_i_inf_at_ti(self.D[1], 0, self.A0), 0)
        self.assertEquals(prob_i_inf_at_ti(self.D[2], 0, self.A0), 0)
        self.assertEquals(prob_i_inf_at_ti(self.D[3], 0, self.A0), 0)
        self.assertEquals(round(prob_i_inf_at_ti(self.D[4], 0, self.A0), 4), 0.0778)
        self.assertEquals(round(prob_i_inf_at_ti(self.D[5], 0, self.A0), 4), 0.2212)
        self.assertEquals(round(prob_i_inf_at_ti(self.D[6], 0, self.A0), 4), 0.0744)

    def test_prob_i_never_infected(self):
        self.assertEquals(round(prob_i_never_infected(self.D[0], 0, self.A0), 4), 0.9)
        self.assertEquals(round(prob_i_never_infected(self.D[1], 0, self.A0), 4), 1.0)
        self.assertEquals(round(prob_i_never_infected(self.D[2], 0, self.A0), 4), 0.975)
        self.assertEquals(round(prob_i_never_infected(self.D[3], 0, self.A0), 4), 0.675)
        self.assertEquals(round(prob_i_never_infected(self.D[4], 0, self.A0), 4), 0.6581)
        self.assertEquals(round(prob_i_never_infected(self.D[5], 0, self.A0), 4), 0.675)
        self.assertEquals(round(prob_i_never_infected(self.D[6], 0, self.A0), 4), 0.6581)

    def test_minus_gamma_hat(self):
        self.assertEquals(round(minus_gamma_hat(self.D[4], 0, self.A0), 4), 2.5543)
        self.assertEquals(round(minus_gamma_hat(self.D[5], 0, self.A0), 4), 1.5088)
        self.assertEquals(round(minus_gamma_hat(self.D[6], 0, self.A0), 4), 2.5988)

    def test_inner_sum_over_Bji_hat(self):
        self.assertEquals(round(inner_sum_over_Bji_hat(self.D[0], 0, self.A0), 4), -0.1054)
        self.assertEquals(round(inner_sum_over_Bji_hat(self.D[1], 0, self.A0), 4), 0)
        self.assertEquals(round(inner_sum_over_Bji_hat(self.D[3], 0, self.A0), 4), -0.3930)

    def test_outer_sum_over_minus_gamma_hat(self):
        # Test that only cascades 1, 2, 4, 5, and 6 are passed
        self.assertEquals(
            outer_sum_over_minus_gamma_hat(self.D, self.A0, 0),
                sum([minus_gamma_hat(self.D[1], 0, self.A0),
                     minus_gamma_hat(self.D[2], 0, self.A0),
                     minus_gamma_hat(self.D[4], 0, self.A0),
                     minus_gamma_hat(self.D[5], 0, self.A0),
                     minus_gamma_hat(self.D[6], 0, self.A0)]))

    def test_outer_sum_over_Bji_hat(self):
        # Test that only cascades 0 and 3 are passed.
        self.assertEquals(
            outer_sum_over_Bji_hat(self.D, self.A0, 0),
                sum([inner_sum_over_Bji_hat(self.D[0], 0, self.A0),
                     inner_sum_over_Bji_hat(self.D[3], 0, self.A0)]))

    def test_convex_formulation(self):
        self.assertEquals(round(convex_formulation(self.A0, i=0, D=self.D), 4), 7.1603)
        self.assertEquals(round(convex_formulation(self.A1, i=1, D=self.D), 4), 9.47)
        self.assertEquals(round(convex_formulation(self.A2, i=2, D=self.D), 4), 2.5264)
        self.assertEquals(round(convex_formulation(self.A3, i=3, D=self.D), 4), 4.9206)

    def test_convex_formulation_with_numpy_array(self):
        self.assertEquals(round(convex_formulation(numpy.array(self.A0), i=0, D=self.D), 4), 7.1603)
        self.assertEquals(round(convex_formulation(numpy.array(self.A1), i=1, D=self.D), 4), 9.47)
        self.assertEquals(round(convex_formulation(numpy.array(self.A2), i=2, D=self.D), 4), 2.5264)
        self.assertEquals(round(convex_formulation(numpy.array(self.A3), i=3, D=self.D), 4), 4.9206)

    def test_minimize(self):
        A_true = numpy.random.rand(4, 4)
        for i in xrange(4):
            for j in xrange(4):
                if i == j:
                    A_true[i][j] = 0
                else:
                    if A_true[i][j] > 0.5:
                        A_true[i][j] = 0.0
                    else:
                        A_true[i][j] *= 0.50

        D = Diffusions(diffusions=[
                Diffusion(A_true) for d in xrange(1000)]
        )

        A_guess = numpy.array(numpy.random.rand(1, 4))

        bounds = [(0, 1) for x in A_guess[0]]
        bounds[0] = (0,0)

        print fmin_tnc(func=convex_formulation,
                       x0=numpy.array(A_guess),
                       args=(0, D),
                       bounds=bounds,
                       approx_grad=True)

        print A_true

class TestDiffusion(ConnieTest):

    def test_where_node_is_infected(self):
        diffusions = [d for d in self.D.where_node_is_infected(0)]
        self.assertEquals(diffusions, [self.D[1], self.D[2], self.D[4], self.D[5], self.D[6]])

        diffusions = [d for d in self.D.where_node_is_infected(1)]
        self.assertEquals(diffusions, [self.D[2], self.D[4], self.D[6]])

        diffusions = [d for d in self.D.where_node_is_infected(2)]
        self.assertEquals(diffusions, [self.D[0], self.D[3], self.D[4], self.D[5], self.D[6]])

        diffusions = [d for d in self.D.where_node_is_infected(3)]
        self.assertEquals(diffusions, [self.D[3], self.D[4], self.D[5], self.D[6]])

    def test_where_node_is_never_infected(self):
        diffusions = [d for d in self.D.where_node_is_never_infected(0)]
        self.assertEquals(diffusions, [self.D[i] for i in [0, 3]])

        diffusions = [d for d in self.D.where_node_is_never_infected(1)]
        self.assertEquals(diffusions, [self.D[i] for i in [0, 1, 3, 5]])

        diffusions = [d for d in self.D.where_node_is_never_infected(2)]
        self.assertEquals(diffusions, [self.D[i] for i in [1, 2]])

        diffusions = [d for d in self.D.where_node_is_never_infected(3)]
        self.assertEquals(diffusions, [self.D[i] for i in [0, 1, 2]])

    def test_node_never_infected(self):
        self.assertTrue(self.D[0].is_never_infected(0))
        self.assertTrue(self.D[0].is_never_infected(1))
        self.assertFalse(self.D[0].is_never_infected(2))
        self.assertTrue(self.D[0].is_never_infected(3))

        self.assertFalse(self.D[1].is_never_infected(0))
        self.assertTrue(self.D[1].is_never_infected(1))
        self.assertTrue(self.D[1].is_never_infected(2))
        self.assertTrue(self.D[1].is_never_infected(3))

        self.assertFalse(self.D[2].is_never_infected(0))
        self.assertFalse(self.D[2].is_never_infected(1))
        self.assertTrue(self.D[2].is_never_infected(2))
        self.assertTrue(self.D[2].is_never_infected(3))

    def test_node_infected(self):
        self.assertFalse(self.D[0].is_infected(0))
        self.assertFalse(self.D[0].is_infected(1))
        self.assertTrue(self.D[0].is_infected(2))
        self.assertFalse(self.D[0].is_infected(3))

        self.assertTrue(self.D[1].is_infected(0))
        self.assertFalse(self.D[1].is_infected(1))
        self.assertFalse(self.D[1].is_infected(2))
        self.assertFalse(self.D[1].is_infected(3))

        self.assertTrue(self.D[2].is_infected(0))
        self.assertTrue(self.D[2].is_infected(1))
        self.assertFalse(self.D[2].is_infected(2))
        self.assertFalse(self.D[2].is_infected(3))

if __name__ == '__main__':
    unittest.main()
