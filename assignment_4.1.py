import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *

# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p

	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]

	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[t_interest, 0]

if __name__ == '__main__':
    ### deterministic setup ###

    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = len(t)/2

    # initial conditions setup
    init_cond   = y0, y1

    ### stochastic setup ####
    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05

    # create uniform distribution object
    distr_w = None

    # the truncation order of the polynomial chaos expansion approximation
    N = [1, 2, 3, 4, 5, 6]
    # the quadrature degree of the scheme used to computed the expansion coefficients
    K = [1, 2, 3, 4, 5, 6]

    # vector to save the statistics
    exp_m = np.zeros(len(N))
    var_m = np.zeros(len(N))

    exp_cp = np.zeros(len(N))
    var_cp = np.zeros(len(N))

    # perform polynomial chaos approximation + the pseudo-spectral
    for h in xrange(len(N)):

        # create N[h] orthogonal polynomials using chaospy
        poly            = None
        # create K[h] quadrature nodes using chaospy
        nodes, weights  = None

        # perform polynomial chaos approximation + the pseudo-spectral approach manually

        # perform polynomial chaos approximation + the pseudo-spectral approach using chaospy
        
    
    print('MEAN')
    print("K | N | Manual \t\t\t| ChaosPy")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=exp_m[h]), '\t|', "{a:1.12f}".format(a=exp_cp[h]))

    print('VARIANCE')
    print("K | N | Manual \t\t| ChaosPy")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=var_m[h]), '\t|', "{a:1.12f}".format(a=var_cp[h]))

