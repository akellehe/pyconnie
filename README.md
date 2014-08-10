# pyconnie

pyconnie is an implementation of Seth A. Myers and Jure Leskovec's approach to inferring latent social networks by exploiting the convexity of the problem.

You can find the paper here: http://arxiv.org/abs/1010.5504

## Example

The example below solves for the 0th column of the adjacency matrix, A with high-precision and low recall.

```python

from scipy.optimize import fmin_tnc 
import numpy

from pyconnie.diffusion import Diffusion, Diffusions
from pyconnie.connie import convex_formulation

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
```
