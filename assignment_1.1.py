import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid):
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, weights, func_eval):
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G 

    return res

def relative_error(approx, ref):
    return np.linalg.norm(approx - ref) / np.linalg.norm(ref)

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

def solve_oscillator(omega): #, t, init_cond, args):
    t = np.arange(0, 10.01, 0.01)  # Time vector from 0 to 10 with step 0.01
    init_cond = [0.5, 0.0]         # Initial conditions y(0) = 0.5 and y'(0) = 0
    args = (0.5, 2.0, 0.5, omega)  # Parameters c, k, f, and omega
    y10 = discretize_oscillator_odeint(model, 1e-8, 1e-8, init_cond, args, t, -1)
    return y10

def chebyshev_grid(a, b, N):
    i = np.arange(1, N + 1)
    x_i = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i - 1) * np.pi / (2 * N))
    return x_i


if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignment
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05
    stat_ref    = [-0.43893703, 0.00019678]

    # create uniform distribution object
    uniform_dist = cp.Uniform(w_left, w_right)

    # no of samples from Monte Carlo sampling
    no_samples_vec = [10, 100, 1000, 10000]
    no_grid_points_vec = [2, 5, 10, 20]

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = -1

    # initial conditions setup
    init_cond   = y0, y1



    # create vectors to contain the expectations and variances and runtimes
    results_expectations = []
    results_variances = []
    runtimes = []

    # compute relative error
    relative_err = lambda approx, real: abs(1. - approx/real)

    results = []

    for no_grid_points in no_grid_points_vec:
        grid = chebyshev_grid(0.95, 1.05, no_grid_points)
        barycentric_weights = compute_barycentric_weights(grid)

        # Evaluate the function at the Chebyshev grid points
        func_eval = np.array([solve_oscillator(omega) for omega in grid])

        for number_of_samples in no_samples_vec:
            omega_samples = uniform_dist.sample(number_of_samples)

            # Evaluate the integral using the barycentric interpolant
            start_time = time.time()
            mc_estimates = []
            for omega_sample in omega_samples:
                interpolated_value = barycentric_interp(omega_sample, grid, barycentric_weights, func_eval)
                mc_estimates.append(interpolated_value)

            mc_estimates = np.array(mc_estimates)
            expected_value = np.mean(mc_estimates)
            variance = np.var(mc_estimates)
            elapsed_time_interp = time.time() - start_time

            print(
                f"N={no_grid_points}, M={number_of_samples}, Expected value: {expected_value}, Variance: {variance}, Time (Interp): {elapsed_time_interp:.4f} seconds")

            # Compare to direct Monte Carlo approach
            start_time = time.time()
            direct_values = np.array([solve_oscillator(omega) for omega in omega_samples])
            direct_expected_value = np.mean(direct_values)
            direct_variance = np.var(direct_values)
            elapsed_time_direct = time.time() - start_time

            # Print direct MC results
            print(
                f"Direct MC M={number_of_samples}, Expected value: {direct_expected_value}, Variance: {direct_variance}, Time (Direct): {elapsed_time_direct:.4f} seconds")

            # Compute relative errors
            relative_error_expected = relative_err(expected_value, stat_ref[0])
            relative_error_variance = relative_err(variance, stat_ref[1])

            relative_error_expected_direct = relative_err(direct_expected_value, stat_ref[0])
            relative_error_variance_direct = relative_err(direct_variance, stat_ref[1])



            # Store results
            results.append((no_grid_points, number_of_samples, expected_value, variance, elapsed_time_interp,
                            direct_expected_value, direct_variance, elapsed_time_direct, relative_error_expected,
                            relative_error_variance, relative_error_expected_direct, relative_error_variance_direct))

    print("\nResults:")
    print(
        "N\tM\tExp. Value (Interp)\tVariance (Interp)\tTime (Interp)\tExp. Value (Direct)\tVariance (Direct)\tTime (Direct)\tRel. Error (Exp. Value)\tRel. Error (Variance)\tRel. Error (Direct Exp. Value)\t(Rel. Error (Direct Variance)")
    for result in results:
        print(
            f"{result[0]}\t{result[1]}\t{result[2]:.6f}\t\t{result[3]:.6f}\t\t{result[4]:.4f}\t{result[5]:.6f}\t\t{result[6]:.6f}\t\t{result[7]:.4f}\t{result[8]:.6f}\t\t\t{result[9]:.6f}\t{result[10]:.6f}\t{result[11]:.6f}")


