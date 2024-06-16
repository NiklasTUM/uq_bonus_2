import chaospy as cp
import numpy as np

if __name__ == '__main__':
	# define the two distributions
	unif_distr = None
	norm_distr = None

	# degrees of the polynomials
	N = [8]	

	# generate orthogonal polynomials for all N's
	for i, n in enumerate(N):
		
		# employ the three terms recursion scheme using chaospy to generate orthonormal polynomials w.r.t. the two distributions

		# compute <\phi_j(x), \phi_k(x)>_\rho, i.e. E[\phi_j(x) \phi_k(x)]

		# print result for specific n
