import chaospy as cp
import numpy as np


def check_orthonormality(polynomials, distribution):
    N = len(polynomials) - 1
    expect = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            expect[i, j] = cp.E(polynomials[i] * polynomials[j], distribution)

    return expect

def verify_result(result_matrix, tolerance=1e-5):
    N = result_matrix.shape[0]
    identity_matrix = np.eye(N)
    return np.allclose(result_matrix, identity_matrix, atol=tolerance)


if __name__ == '__main__':
    # define the two distributions
    unif_distr = cp.Uniform(-1, 1)
    norm_distr = cp.Normal(10, 1)

    # degrees of the polynomials
    N = 8

    uniform_dist = cp.Uniform(-1, 1)
    uniform_poly = cp.generate_expansion(N, uniform_dist, normed=True)
    result_matrix_uniform = check_orthonormality(uniform_poly, uniform_dist)
    print(verify_result(result_matrix_uniform))

    normal_dist = cp.Normal(10, 1)
    normal_poly = cp.generate_expansion(N, normal_dist, normed=True)
    result_matrix_normal = check_orthonormality(normal_poly, normal_dist)
    print(result_matrix_normal)
    print(verify_result(result_matrix_normal))


