#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This code implements DR algorithms to reduce the dimensionality of a data set, as well as DR quality assessment criteria.
# Pay attention to the fact that neighbor embedding DR methods do not like data sets with repeated samples.

# References:
#
# [1] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
# [2] Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
# [3] Jacobs, R. A. (1988). Increased rates of convergence through learning rate adaptation. Neural networks, 1(4), 295-307.
# [4] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
# [5] Hinton, G., & Roweis, S. (2002, January). Stochastic neighbor embedding. In NIPS (Vol. 15, pp. 833-840).
# [6] Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010). Information retrieval perspective to nonlinear dimensionality reduction for data visualization. Journal of Machine Learning Research, 11(Feb), 451-490.
# [7] John A. Lee, Michel Verleysen.
#     Rank-based quality assessment of nonlinear dimensionality reduction.
#     Proc. 16th ESANN, Bruges, pp 49-54, April 2008.
# [8] John A. Lee, Michel Verleysen.
#     Quality assessment of nonlinear dimensionality reduction: 
#     rank-based  criteria.
#     Neurocomputing, 72(7-9):1431-1443, March 2009.
# [9] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
# [10] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# [11] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2014). Multiscale stochastic neighbor embedding: Towards parameter-free dimensionality reduction. In ESANN.
# [12] Carreira-Perpinán, M. A. (2010, June). The Elastic Embedding Algorithm for Dimensionality Reduction. In ICML (Vol. 10, pp. 167-174).
# [13] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing , 16, 5, pp. 1190-1208.
# [14] Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. Journal of machine learning research, 15(1), 3221-3245.
# [15] Demartines, P., & Hérault, J. (1997). Curvilinear component analysis: A self-organizing neural network for nonlinear mapping of data sets. IEEE Transactions on neural networks, 8(1), 148-154.
# [16] Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.
# [17] Yang, Z., Peltonen, J., & Kaski, S. (2013, June). Scalable Optimization of Neighbor Embedding for Visualization. In ICML (2) (pp. 127-135).

# author: Cyril de Bodt (ICTEAM - UCL)
# @email: cyril_debodt __at__ uclouvain.be
# Last modification date: March 6, 2018
# Copyright (c) 2018 Universite catholique de Louvain, ICTEAM. All rights reserved.

# This code was created and tested on Python 3.5.2 (Anaconda distribution, Continuum Analytics, Inc.) with the following modules
# - NumPy: verion 1.11.2 
# - Scikit-learn: version 0.18.1 
# - Scipy: version 0.18.1
# - numba: version 0.34.0

# You can use and modify this software freely, but not for commercial purposes. 
# The use of the software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

########################################################################################################
########################################################################################################

import numpy as np, copy, sklearn.decomposition, scipy.special, numba, scipy.optimize, scipy.spatial.distance, sklearn.datasets, matplotlib.pyplot as plt


##############################
############################## Global variables
##############################

# Module name
module_name = 'multi_scale'
# Maximum number of iterations in the gradient-based optimization procedures. 
sim_dr_nitmax = 100000
# Tolerance on the gradient infinite norm in gradient-based optimization procedures. 
sim_dr_gtol = 10**(-5)
# Tolerance on the cost function relative updates in gradient-based optimization procedures. 
sim_dr_ftol = 2.2204460492503131e-09
# Maximum number of line search steps per L-BFGS iteration.
sim_dr_maxls = 50
# The maximum number of variable metric corrections used to define the limited memory matrix in L-BFGS. 
sim_dr_maxcor = 10

n_eps_np_float64 = np.finfo(dtype=np.float64).eps

##############################
############################## General functions
##############################


def compute_rel_obj_diff(prev_obj_value, cur_obj_value):
    """
    Compute the relative objective function difference between two steps in a gradient descent.
    In: 
    - prev_obj_value: previous objective function value.
    - cur_obj_value: current objective function value.
    Out:
    np.abs(prev_obj_value - cur_obj_value)/max(np.abs(prev_obj_value), np.abs(cur_obj_value))
    """
    return np.abs(prev_obj_value - cur_obj_value)/np.maximum(np.finfo(dtype=np.float64).eps, max(np.abs(prev_obj_value), np.abs(cur_obj_value)))

def compute_grad_norm(grad_lds):
    """
    Compute the norm of the gradient for the stopping criterion of the gradient descent process.
    In: 
    - grad_lds: current gradient value.
    Out:
    Infinite norm of the gradient.
    """
    return np.absolute(grad_lds).max()

def handle_perp(perp, N):
    """
    Manage the perplexity value.
    In:
    - perp: perplexity value. If <= 0, an error is raised.
    - N: data set size.
    Out:
    An integer v being the perplexity. If perp is in ]0,1[, then v is first set to perp*N. Otherwise, v is set to perp. Then, v is rounded. Then if v is <= 1.0, it is set to 2.0. 
    """
    global module_name
    if perp <= 0:
        raise ValueError("Error in function handle_perp of module {module_name}: perp={perp}, which is negative.".format(module_name=module_name, perp=perp))
    elif perp < 1.0:
        perp *= np.float64(N)
    perp = np.around(perp)
    if perp <= 1.0:
        perp = 2.0
    return np.int64(perp)

@numba.jit(nopython=True)
def arange_except_i(N, i):
    """
    Create a one-dimensional numpy array of integers from 0 to N with step 1, except i.
    In:
    - N: a strictly positive integer.
    - i: a positive integer which is strictly smaller than N.
    Out:
    A one-dimensional numpy array of integers from 0 to N with step 1, except i.
    """
    arr = np.arange(N)
    return np.hstack((arr[:i], arr[i+1:]))

@numba.jit(nopython=True)
def close_to_zero(v):
    """
    Check whether v is close to zero or not.
    In:
    - v: a scalar or numpy array.
    Out:
    A boolean or numpy array of the same shape as v, with True when the entry is close to 0 and False otherwise.
    """
    return np.absolute(v)<=10.0**(-8.0)

def pairwise_dist_tomatrix(d):
    """
    Transforms a vector of pairwise distances computed by function pairwise_dist into a redundant matrix of pairwise distances.
    In:
    - d: vector of pairwise distances, as computed by pairwise_dist with tomatrix=False.
    Out:
    A redundant matrix of pairwise distances.
    """
    return scipy.spatial.distance.squareform(X=d, force='tomatrix')

def pairwise_dist(X, metric='euclidean', tomatrix=False):
    """
    Computes the pairwise distances between the examples in a data set.
    In:
    - X: a numpy.ndarray representing a data set with one feature per column and one example per row.
    - metric: the metric to use to compute the distances.
    - tomatrix: boolean. If True, returns the redundant matrix of pairwise distances. Must be False by default.
    Out:
    If tomatrix is False:
        A one dimensional numpy array returned by scipy.spatial.distance.pdist(X=X, metric=metric). This vector contains the distances between all the distinct pairs of examples in X. This representation saves memory compared to the matrix representation with redundant elements.
        To extract to distances between the examples in the rows i and j in X, you can use the function get_dist.
    If tomatrix is True:
        The redundant matrix of pairwise distances.
    """
    d = scipy.spatial.distance.pdist(X=X, metric=metric)
    if tomatrix:
        d = pairwise_dist_tomatrix(d=d)
    return d

def np_inf2v(arr, v):
    """
    Convert infinite values in a numpy array arr to some given value v.
    This function returns nothing. It modifies arr.
    In:
    - arr: numpy array.
    - v: a value, with type compatible with the one of arr.
    """
    arr[np.logical_not(np.isfinite(arr))] = v

@numba.jit(nopython=True)
def fill_diago(M, v):
    """
    Replaces the elements on the diagonal of a squared matrix by some value.
    In:
    - M: a two-dimensional numpy array containing a squared matrix.
    - v: a value of the same type as M.
    Out:
    M by in which the diagonal elements have been replaced by v.
    """
    for i in range(M.shape[0]):
        M[i,i] = v
    return M

def lbfgsb(fun, x0, args=(), jac=None, disp=False, gtol=1e-05, maxiter=15000, bounds=None, maxls=20, callback=None, maxcor=10, maxfun=np.inf, ftol=2.2204460492503131e-09):
    """
    Minimization of scalar function of one or more variables using the L-BFGS-B algorithm.
    
    In:
    fun : callable
        Objective function (scalar, multivariate). It must take a one-dimensional array as argument.
    x0 : ndarray
        Initial guess. One-dimensional numpy array.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives (Jacobian, Hessian).
    jac : bool or callable, optional
        Jacobian (gradient) of objective function. Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function. If False, the gradient will be estimated numerically. jac can also be a callable returning the gradient of the objective. In this case, it must accept the same arguments as fun.
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    bounds : sequence, optional
        Bounds for variables (only for L-BFGS-B, TNC and SLSQP). (min, max) pairs for each element in x, defining the bounds on that parameter. Use None for one of min or max when there is no bound in that direction.
    gtol : float
        Gradient norm must be less than gtol before 
    maxls: int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current parameter vector.
    maxcor : int
        The maximum number of variable metric corrections used to define the limited memory matrix. (The limited memory BFGS method does not store the full hessian but uses this many terms in an approximation to it.)
    maxfun: int
        Maximum number of function evaluations.
    ftol : float
        The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
    
    See 
    - https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    - https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    - https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_minimize.py#L36-L466
    - https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/optimize.py
    (last consulted on December 20th, 2016) for more documentation.
    
    """
    return scipy.optimize.minimize(fun=fun, x0=x0, args=args, method='L-BFGS-B', jac=jac, bounds=bounds, callback=callback, options={'disp':disp, 'maxls':maxls, 'gtol':gtol, 'maxiter':maxiter, 'maxcor':maxcor, 'maxfun':maxfun, 'ftol':ftol})

def delta_bar_delta_rule(delta_bar, grad, stepsize, kappa=0.2, phi=0.8, theta_delta_bar=0.5):
    """
    Apply the delta-bar-delta stepsize adaptation rule in a gradient descent procedure.
    In:
    - delta_bar: numpy array with as many components as variables in the considered optimization problem, and which stores the current value of delta_bar defined in [3]. The latter should be initialized with zero's at the beginning of the process.
    - grad: numpy array with as many components as variables in the considered optimization problem, and which stores the value of gradient at current coordinates.
    - stepsize: numpy array with as many components as variables in the considered optimization problem, and which stores the current value of the stepsize associated with each variables.
    - kappa: linear stepsize increase (for a variable) when delta_bar and the gradient are of the same sign.
    - phi: exponential stepsize decrease (for a variable) when delta_bar and the gradient are of different signs.
    - theta_delta_bar: parameter for the update of delta_bar.
    Out:
    A tuple with two elements:
    - A numpy array with the update of delta_bar.
    - A numpy array with the update of the stepsize.
    """
    delta_bar_delta_prod = np.sign(delta_bar) * np.sign(grad)
    stepsize[delta_bar_delta_prod > 0] += kappa
    stepsize[delta_bar_delta_prod < 0] *= phi
    delta_bar = (1-theta_delta_bar) * grad + theta_delta_bar * delta_bar
    return delta_bar, stepsize

def momentum_gradient_descent_step(X, update_X, nit, mom_t_change, mom_init, mom_fin, stepsize, grad):
    """
    Performs a momentum gradient step in a gradient descent procedure with momentum.
    In:
    - X: numpy array containing the current value of the variables in the gradient descent procedure.
    - update_X: numpy array with the same shape as X storing the previous update made on the variables at the previous gradient step.
    - nit: number of gradient iterations which have already been performed.
    - mom_t_change: number of gradient steps to perform before changing the momentum coefficient.
    - mom_init: momentum coefficient to use when nit<mom_t_change.
    - mom_fin: momentum coefficient to use when nit>=mom_t_change.
    - stepsize: step size to use in the gradient descent. Either a scalar, or a numpy array with the same dimension as X.
    - grad: numpy array with the same shape as X, storing the gradient of the objective function at the current coordinates.
    Out:
    A tuple with two elements:
    - A numpy array with the updated coordinates, after having performed the momentum gradient descent step.
    - A numpy array storing the update performed on the variables.
    """
    if nit < mom_t_change:
        mom = mom_init
    else:
        mom = mom_fin
    update_X = mom * update_X - (1-mom) * stepsize * grad
    X += update_X
    return X, update_X

##############################
############################## PCA. 
##############################

def apply_pca(X_hds, n_components, whiten):
    """
    Apply PCA on a data set X_hds with a set of hyper-parameters.
    In:
    - X_hds: a 2-dimensional numpy.ndarray, containing one example per row and one feature per column.
    - n_components, whiten: parameter of the function sklearn.decomposition.PCA. See the online documentation for explanations.
    Out:
    The low dimensional representation of X_hds. It contains one example per row and one feature per column. Example in row i corresponds to the example in row i of X_hds.
    """
    # Defining the method
    meth = sklearn.decomposition.PCA(whiten=whiten, copy=True, svd_solver='full', iterated_power='auto', tol=0.0, random_state=0)
    # Returning
    return meth.fit_transform(X_hds)[:,:n_components]

##############################
############################## SNE (Stochastic neighbor embedding). When adjusting the HDS similarities, if the perplexity is greater than 0.99*N (where N is the number of examples in the data set), it is decreased toward min(perp, np.floor(0.99*N)).
##############################

@numba.jit(nopython=True)
def sne_hds_p_i_js(ds_hds_i_js, den_s_i, i, compute_log=True):
    """
    Computes the SNE asymmetric HDS probabilities p_{ij}'s for j = 0...N-1, with N being the number of data points, as defined in [5], as well as their log. i and j are assumed to be indexes ranging from 0 to N-1 (included).
    In:
    - ds_hds_i_js: numpy one-dimensional array with N squared distances in HDS. Element k is the squared HDS distance between k and i in HDS.
    - den_s_i: a scalar being equal to 2*(sigma_i**2). This is the denominator in the exponentials of p_{ij}'s for all j's.
    - i: index of i.
    - compute_log: boolean. If true, the logarithms of the p_{ij}'s are computed. Otherwise not.
    Out:
    A tuple with two elements:
    - A numpy array with N elements. Element k is p_{ik}, as defined in [5]. p_[ii] is set to 0.
    - If compute_log is True, a numpy array with N element. Element k is log(p_{ik}). By convention, log(p_{ii}) is set to 0. If compute_log is False, it is set to np.empty(shape=(N), dtype=np.float64).
    """
    # Number of data points
    N = ds_hds_i_js.size
    # Initializing the arrays with the p_{ij}'s and the log(p_{ij})'s.
    p_js_i = np.empty(shape=N, dtype=np.float64)
    p_js_i[i] = 0.0
    log_p_js_i = np.empty(shape=N, dtype=np.float64)
    if compute_log:
        log_p_js_i[i] = 0.0
    # Indexes of the data points, exept id
    idx_js_no_i = arange_except_i(N=N, i=i)
    # Squared distances between i and the other datapoints
    ds_hds_i_js_no_i = ds_hds_i_js[idx_js_no_i]
    # Arguments of the exponentials at the numerator of the similarities. 
    log_num_p_js_i_no_i = (ds_hds_i_js_no_i.min()-ds_hds_i_js_no_i)/den_s_i
    # Numerators of the similarities
    p_js_i[idx_js_no_i] = np.exp(log_num_p_js_i_no_i)
    # Denominator of the similarities. 
    den_p_js_i = p_js_i.sum()
    # Computing the similarities. 
    p_js_i /= den_p_js_i
    # Computing the log of the similarities
    if compute_log:
        log_p_js_i[idx_js_no_i] = log_num_p_js_i_no_i - np.log(den_p_js_i)
    return p_js_i, log_p_js_i

@numba.jit(nopython=True)
def sne_entropy_p_i_js(ds_hds_i_js, den_s_i, i):
    """
    Computes the entropy of the SNE asymmetric HDS probabilities p_{ij} for j = 0...N-1, with N being the number of data points, as defined in [5]. i and j are assumed to be indexes ranging from 0 to N-1 (included).
    In:
    - ds_hds_i_js: numpy one-dimensional array with N squared distances in HDS. Element k is the squared HDS distance between k and i in HDS.
    - den_s_i: a scalar being equal to 2*(sigma_i**2). This is the denomintor in the exponentials of p_{j|i}'s for all j's.
    - i: index of i.
    Out:
    The entropy of the p_{ij} for j = 0...N-1.
    """
    p_js_i, log_p_js_i = sne_hds_p_i_js(ds_hds_i_js=ds_hds_i_js, den_s_i=den_s_i, i=i, compute_log=True)
    return -np.dot(p_js_i, log_p_js_i)

@numba.jit(nopython=True)
def sne_bin_search_fct(ds_hds_i_js, den_s_i, i, log_perp):
    """
    Evaluates the function on which a binary search is performed to find the bandwidth of the i^th data point in SNE.
    In: 
    - ds_hds_i_js, den_s_i, i: same as in sne_entropy_p_i_js
    - log_perp: logarithm of the perplexity.
    Out:
    A float being sne_entropy_p_i_js(ds_hds_i_js=ds_hds_i_js, den_s_i=den_s_i, i=i) - log_perp
    """
    return sne_entropy_p_i_js(ds_hds_i_js=ds_hds_i_js, den_s_i=den_s_i, i=i) - log_perp

@numba.jit(nopython=True)
def sne_bin_search_perp_fit(ds_hds_i_js, i, log_perp, x0=1.0):
    """
    Perform a binary seach to find the root of sne_bin_search_fct over its scalar argument den_s_i. 
    In:
    - ds_hds_i_js, i, log_perp: same as in function sne_bin_search_fct.
    - x0: starting point for the binary search. Must be strictly positive.
    Out:
    A strictly positive float den_s_i such that sne_bin_search_fct(ds_hds_i_js, den_s_i, i, log_perp) is close to zero, except if the binary search has failed. The latter happens when the root is so close to 0 that sne_bin_search_fct becomes undefined, as it is only defined for den_s_i>0. In this case, a non-root den_s_i of sne_bin_search_fct yielding its smallest found value is returned, and a warning message is printed.
    """
    # Find x_up and x_low such that sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_up, i=i, log_perp=log_perp) > 0 and sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_low, i=i, log_perp=log_perp) < 0
    fx0 = sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x0, i=i, log_perp=log_perp)
    if close_to_zero(v=fx0):
        return x0
    elif not np.isfinite(fx0):
        raise ValueError("Error in function sne_bin_search_perp_fit of module leonardo_dr: fx0 is nan.")
    elif fx0 > 0:
        x_up, x_low = x0, x0/2.0
        fx_low = sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_low, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_low):
            return x_low
        elif not np.isfinite(fx_low):
            # WARNING: can not find a valid root!
            return x_up
        while fx_low > 0:
            x_up, x_low = x_low, x_low/2.0
            fx_low = sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_low, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_low):
                return x_low
            # If sne_bin_search_fct becomes undefined because x_low is too near from 0, then return x_up. Binary search has failed.
            if not np.isfinite(fx_low):
                return x_up
    else: # fx0 < 0
        x_up, x_low = x0*2.0, x0
        fx_up = sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_up, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_up):
            return x_up
        elif not np.isfinite(fx_up):
            # WARNING: can not find a valid root!
            return x_low
        while fx_up < 0:
            x_up, x_low = 2.0*x_up, x_up
            fx_up = sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_up, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_up):
                return x_up
    # sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_up, i=i, log_perp=log_perp) > 0 and sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x_low, i=i, log_perp=log_perp) < 0 => Start binary search
    while True:
        x = (x_up+x_low)/2.0
        fx = sne_bin_search_fct(ds_hds_i_js=ds_hds_i_js, den_s_i=x, i=i, log_perp=log_perp)
        if close_to_zero(v=fx):
            return x
        elif fx > 0:
            x_up = x
        else: # fx < 0
            x_low = x

@numba.jit(nopython=True)
def sne_hds_similarities(dsm_hds, perp, compute_log=True, start_bs=np.ones(shape=1, dtype=np.float64)):
    """
    Computes the matrix of SNE asymmetric probabilities in HDS, as defined in [5], as well as their log.
    In:
    - dsm_hds: two-dimensional numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HDS distance between i and j.
    - perp: perplexity. Must be > 1.
    - compute_log: boolean. If true, the logarithms of the p_{ij}'s are computed. Otherwise not.
    - start_bs: one-dimensional numpy array with N elements. Element at index i is the starting point of the binary search for the ith data point. If start_bs has only one element, it will be set to np.ones(shape=N, dtype=np.float64).
    Out:
    A tuple with three elements:
    - A two-dimensional numpy array with shape (N, N) and in which element [i,j] = p_{ij}. p_{ii} is set to 0.
    - If compute_log is True, two-dimensional numpy array with shape (N, N) and in which element [i,j] = log(p_{ij}). By convention, log(p_{ii}) is set to 0. If compute_log is False, it is set to np.empty(shape=(N,N), dtype=np.float64).
    - A one-dimensional numpy array with N elements, where element i is the denominator of the exponentials of the p_{ij}'s for j=0, ..., N-1 and for some i. Hence element i is equal to 2*(sigma_i**2).
    """
    if perp <= 1:
        raise ValueError("""Error in function sne_hds_similarities of module leonardo_dr.py: the perplexity should be >1.""")
    N = dsm_hds.shape[0]
    if start_bs.size == 1:
        start_bs = np.ones(shape=N, dtype=np.float64)
    log_perp = np.log(min(np.float64(perp), np.floor(0.99*np.float64(N))))
    # Computing the N**2 HDS similarities p_{ij}, for i, j = 0, ..., N-1. p_ij[i,j] contains p_{ij}
    p_ij = np.empty(shape=(N,N), dtype=np.float64)
    log_p_ij = np.empty(shape=(N,N), dtype=np.float64)
    arr_den_s_i = np.empty(shape=N, dtype=np.float64)
    for i in range(N):
        # Computing 2*(sigma_i**2). 
        den_s_i = sne_bin_search_perp_fit(ds_hds_i_js=dsm_hds[i,:], i=i, log_perp=log_perp, x0=start_bs[i])
        # Computing p_{ij} for j=0, ..., N-1.
        tmp = sne_hds_p_i_js(ds_hds_i_js=dsm_hds[i,:], den_s_i=den_s_i, i=i, compute_log=compute_log)
        p_ij[i,:] = tmp[0]
        if compute_log:
            log_p_ij[i,:] = tmp[1]
        arr_den_s_i[i] = den_s_i
    return p_ij, log_p_ij, arr_den_s_i

def sne_lds_similarities_fast(dsm_lds, arr_den_s_i=None, compute_log=True):
    """
    Computation of the matrix of SNE asymmetric probabilities in LDS, as defined in [5], as well as their log.
    In:
    - dsm_lds: two-dimensional numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared LDS distance between i and j.
    - arr_den_s_i: numpy one-dimensional array with N elements, where element i is the denominator of the exponentials of the p_{ij}'s for some i and for j=0, ..., N-1. Hence element i is equal to 2*(sigma_i**2). If equal to None, it is set to np.ones(shape=N).
    - compute_log: boolean. If true, the logarithms of the q_{ij}'s are computed. Otherwise not.
    Out:
    A tuple with two elements:
    - A two-dimensional numpy array with shape (N, N) and in which element [i,j] = q_{ij}. q_{ii} is set to 0.
    - If compute_log is True, two-dimensional numpy array with shape (N, N) and in which element [i,j] = log(q_{ij}). By convention, log(q_{ii}) is set to 0. If compute_log is False, None.
    """
    N = dsm_lds.shape[0]
    if arr_den_s_i is None:
        arr_den_s_i = np.ones(shape=N)
    # Extracting the values on the diagonal of dsm_lds for the search of the smallest nonzero distance for each dot
    diago_dsm_lds = np.diagonal(a=dsm_lds, offset=0, axis1=0, axis2=1).copy()
    # Modifying the values of the diagonal of dsm_lds for the search of the smallest non-diagonal distance for each dot
    np.fill_diagonal(a=dsm_lds, val=np.inf)
    # Arguments of the exponentials at the numerator of the similarities
    log_num_q_js_i = ((dsm_lds.min(axis=1)-dsm_lds.T)/np.maximum(np.finfo(dtype=np.float64).eps, arr_den_s_i)).T
    # Numerators of the similarities
    q_ij = np.exp(log_num_q_js_i)
    # Correcting the diagonal of the similarities
    np.fill_diagonal(a=q_ij, val=0)
    # Denominator of the similarities. 
    den_q_ij = np.dot(q_ij, np.ones(shape=N))
    # Computing the N**2 LDS similarities q_{ij}, for i, j = 0, ..., N-1. q_ij[i,j] contains q_{ij}
    q_ij = (q_ij.T/np.maximum(np.finfo(dtype=np.float64).eps, den_q_ij)).T
    # Computing the log of the similarities
    if compute_log:
        log_q_ij = (log_num_q_js_i.T - np.log(den_q_ij)).T
        # Correcting the diagonal
        np.fill_diagonal(a=log_q_ij, val=0)
    else:
        log_q_ij = None
    # Putting back the value of the diagonal of dsm_lds.
    np.fill_diagonal(a=dsm_lds, val=diago_dsm_lds)
    return q_ij, log_q_ij

def sne_init_random_embedding(N, n_components, rand_state):
    """
    Initialize the low dimensional embedding randomly.
    In:
    - N: number of samples
    - n_components: Number of dimensions in the LDS
    - rand_state: random state
    Out:
    A numpy.ndarray with shape (N, 2), containing the initialization of the low dimensional data set, with one row per example and one line per dimension.
    """
    return 10**(-4) * rand_state.randn(N, n_components)

def sne_init_embedding(X_hds, init, n_components, rand_state):
    """
    Initialize the low dimensional embedding.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the high dimensional data set, with one row per example and one line per dimension.
    - init: string indicating which type of initialization to use for the low dimensional embedding. If equal to 'pca', the low dimensional embedding is initialized with the first two principal components of X_hds. It equal to 'random', it is initialized to 10**(-4) * rand_state.randn(N, n_components). If equal to another value, a ValueError error is raised.
    - n_components: Number of dimensions in the LDS
    - rand_state: random state
    Out:
    A numpy.ndarray with shape (N, 2), containing the initialization of the low dimensional data set, with one row per example and one line per dimension.
    """
    global module_name
    if init == "pca":
        # Apply a PCA and keep the first n_components principal components.
        return apply_pca(X_hds=X_hds, n_components=n_components, whiten=False)
    elif init == 'random':
        return sne_init_random_embedding(N=X_hds.shape[0], n_components=n_components, rand_state=rand_state)
    else:
        raise ValueError("Error in function sne_init_embedding of module {module_name}: unknown value '{init}' for init parameter.".format(module_name=module_name, init=init))

def sne_obj_fct(p_ij, log_p_ij, log_q_ij):
    """
    Computes the KL_divergence between the similarities in HDS and in LDS. This function correctly handles the zero's in p_ij, log_p_ij and log_q_ij.
    In:
    - p_ij: two-dimensional numpy array, in which element [i,j] contains the HDS pairwise similarity between i and j, defined as in [5].
    - log_p_ij: two-dimensional numpy array, in which element [i,j] contains the natural logarithm of the HDS pairwise similarity between i and j, defined as in [5].
    - log_q_ij: two-dimensional numpy array, in which element [i,j] contains the natural logarithm of the LDS pairwise similarity between i and j, defined as in [5].
    Out:
    The KL divergence between p_ij and q_ij = sum_{k,l=0, ..., N-1} p_ij[k,l]*np.log(p_ij[k,l]/q_ij[k,l]), where N is the number of data points.
    """
    return np.dot(p_ij.flatten(), (log_p_ij-log_q_ij).flatten())

def sne_grad_cur_obj(X_lds, p_ij, log_p_ij, nit, early_exa_it_stop, early_exa_factor):
    """
    Computes the gradient of the objective function of SNE at some LDS coordinates, as well as the current value of the objective function.
    In:
    - X_lds: two-dimensional numpy array, with shape (N, 2), where N is the number of data points. It contains one example per row and one feature per column. it stores the current LDS coordinates.
    - p_ij: two-dimensional numpy array, with shape (N, N), where element [i,j] contains p_{ij}, as defined in [5].
    - log_p_ij: two-dimensional numpy array, with shape (N, N), where element [i,j] contains log(p_{ij}), as defined in [5]. By convention, log(p_{ii}) must be equal to 0.
    - nit: Number of gradient steps which have already been performed.
    - early_exa_it_stop: number of gradient steps to perform with early exageration.
    - early_exa_factor: early exageration factor.
    Out:
    A tuple with two elements:
    - grad_lds: a two-dimensional numpy array, with shape (N, 2), containing the gradient at X_lds.
    - cur_obj_value: current objective function value at X_lds.
    """
    # Computing the similarities in the LDS.  To compute the LDS distances, we use "sqeuclidean" metric instead of "euclidean" to avoid squaring the distances.
    q_ij, log_q_ij = sne_lds_similarities_fast(dsm_lds=pairwise_dist(X=X_lds, metric='sqeuclidean', tomatrix=True))
    
    # Computing the current objective function value
    if nit < early_exa_it_stop:
        # Removing the early exageration to compute the objective function value.
        cur_obj_value = sne_obj_fct(p_ij=p_ij/early_exa_factor, log_p_ij=log_p_ij-np.log(early_exa_factor), log_q_ij=log_q_ij)
    else:
        cur_obj_value = sne_obj_fct(p_ij=p_ij, log_p_ij=log_p_ij, log_q_ij=log_q_ij)
    
    # Computing multiplication factor in the gradient
    c_ij = 2*(p_ij+p_ij.T-q_ij-q_ij.T)
    # Computing the gradient. 
    grad_lds = (X_lds.T*np.dot(c_ij, np.ones(shape=X_lds.shape[0]))).T - np.dot(a=c_ij, b=X_lds)
    # Returning
    return grad_lds, cur_obj_value

def sne_implem(X_hds, perp, init):
    """
    This function applies SNE to reduce the dimensionality of a data set to 2 dimensions.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the high dimensional data set, with one row per example and one column per dimension.
    - perp: perplexity. If perp<=1, an error is raised.
    - init: see function sne_init_embedding for a description.
    Out:
    A numpy.ndarray with shape (N, 2), containing the low dimensional data set, with one row per example and one column per dimension.
    Remarks:
    - Gradient descent with momentum is applied, as suggested in [2, 3]. The step size is adapted using the delta-bar-delta rule introduced in [3], as indicated in [2].
    - Random gaussian jitter that decreases with time is added to the low dimensional coordinates, as in [5].
    - At most nit_max gradient steps are computed. The iterations stops as soon as the infinite norm of the gradient is small enough or as the objective function is not evolving anymore.
    - The LD coordinates reaching the lowest objective function value among all the ones obtained throughout the iterations are returned.
    """
    global sim_dr_nitmax, sim_dr_gtol, sim_dr_ftol
    
    # Defining the random state
    rand_state = np.random.RandomState(0)
    # Number of data points
    N = X_hds.shape[0]
    # Dimension of the LDS
    n_components = 2
    # Maximum number of gradient descent iterations.
    nit_max = sim_dr_nitmax
    # Current number of gradient descent iterations
    nit = 0
    # Tolerance for the norm of the gradient in the gradient descent algorithm
    gtol = sim_dr_gtol
    # Tolerance for the relative update of the value of the objective function.
    ftol = sim_dr_ftol
    
    # Initializing the low dimensional embedding
    X_lds = sne_init_embedding(X_hds=X_hds, init=init, n_components=n_components, rand_state=rand_state)
    
    # Computing the HDS similarities. We use "sqeuclidean" metric instead of "euclidean" to avoid squaring the distances.
    p_ij, log_p_ij = sne_hds_similarities(dsm_hds=pairwise_dist(X=X_hds, metric='sqeuclidean', tomatrix=True), perp=perp, compute_log=True)[:2]
    
    # Early exageration factor
    early_exa_factor = 4
    # Number of gradient steps to perform with early exageration. None are performed in this case, as it has been observed to provide better results.
    early_exa_it_stop = 0
    if early_exa_it_stop > nit:
        p_ij *= early_exa_factor
        log_p_ij += np.log(early_exa_factor)
    
    # Computing the current gradient and objective function values.
    grad_lds, cur_obj_value = sne_grad_cur_obj(X_lds=X_lds, p_ij=p_ij, log_p_ij=log_p_ij, nit=nit, early_exa_it_stop=early_exa_it_stop, early_exa_factor=early_exa_factor)
    grad_norm = compute_grad_norm(grad_lds=grad_lds)
    # Low dimensional coordinates achieving the smallest reached value of the objective function.
    best_X_lds = copy.deepcopy(X_lds)
    # Smallest reached value of the objective function
    best_obj_value = cur_obj_value
    # Previous objective function value. This value is chosen in order to enter the gradient descent loop.
    prev_obj_value = (1+100*ftol)*cur_obj_value
    rel_obj_diff = compute_rel_obj_diff(prev_obj_value=prev_obj_value, cur_obj_value=cur_obj_value)
    
    # Momentum parameters
    mom_init, mom_fin, mom_t_change = 0.5, 0.8, 250
    
    # Step size parameters. Each optimization variable (i.e. LDS coordinate) has its own step size. The steps are adapted during the gradient descent using the Delta-Bar-Delta learning rule, from [3].
    epsilon, kappa, phi, theta_delta_bar = 1, 0.2, 0.8, 0.5
    stepsize, delta_bar = epsilon*np.ones(shape=(N, n_components), dtype=np.float64), np.zeros(shape=(N, n_components), dtype=np.float64)
    
    # Update of X_lds
    update_X_lds = np.zeros(shape=(N, n_components), dtype=np.float64)
    
    # Gradient descent. The iterations either stops if the maximum number of iteration allowed is reached, or if the infinite norm of the gradient is small enough or if the objective function is not evolving anymore.
    while (nit < nit_max) and (grad_norm > gtol) and (rel_obj_diff > ftol):
        # Computing the step sizes, following the delta-bar-delta rule, from [3].
        delta_bar, stepsize = delta_bar_delta_rule(delta_bar=delta_bar, grad=grad_lds, stepsize=stepsize, kappa=kappa, phi=phi, theta_delta_bar=theta_delta_bar)
        
        # Performing the gradient descent step with momentum
        X_lds, update_X_lds = momentum_gradient_descent_step(X=X_lds, update_X=update_X_lds, nit=nit, mom_t_change=mom_t_change, mom_init=mom_init, mom_fin=mom_fin, stepsize=stepsize, grad=grad_lds)
        # Centering the result
        X_lds -= X_lds.mean(axis=0)
        
        # Adding random jitter that decrease with time
        X_lds += rand_state.randn(N, n_components)*np.exp(-nit/2.0)
        
        # Incrementing the iteration counter
        nit += 1
        # Checking whether or not early exageration is over
        if nit == early_exa_it_stop:
            p_ij /= early_exa_factor
            log_p_ij -= np.log(early_exa_factor)
        
        # Updating the previous objective function value
        prev_obj_value = cur_obj_value
        # Computing the gradient at the current LD coordinates and the current objective function value.
        grad_lds, cur_obj_value = sne_grad_cur_obj(X_lds=X_lds, p_ij=p_ij, log_p_ij=log_p_ij, nit=nit, early_exa_it_stop=early_exa_it_stop, early_exa_factor=early_exa_factor)
        grad_norm = compute_grad_norm(grad_lds=grad_lds)
        rel_obj_diff = compute_rel_obj_diff(prev_obj_value=prev_obj_value, cur_obj_value=cur_obj_value)
        
        # Updating best_obj_value and best_X_lds
        if best_obj_value > cur_obj_value:
            best_obj_value, best_X_lds = cur_obj_value, copy.deepcopy(X_lds)
    
    # Returning
    return best_X_lds

@numba.jit(nopython=True)
def ms_perplexities(N, K_star=2):
    """
    Define exponentially growing multi-scale perplexities, as defined in [10].
    In:
    - N: number of data points.
    - K_star: K_{*} as defined in [10] to set the multi-scale perplexities.
    Out:
    A tuple with 4 elements:
    - L_min, as defined in [10]
    - L_max, as defined in [10]
    - L, as defined in [10]
    - K_h: one-dimensional numpy array, with the perplexities in increasing order.
    """
    # Defining L_min and L_max
    L_min, L_max = 1, int(round(np.log2(np.float64(N)/np.float64(K_star))))
    # Number of considered perplexities
    L = L_max-L_min+1
    # Array with the considered perplexities.
    K_h = (np.float64(2.0)**(np.linspace(L_min-1, L_max-1, L).astype(np.float64)))*np.float64(K_star)
    # Returning
    return L_min, L_max, L, K_h


##############################
############################## Multiscale SNE, from [10]. 
##############################

@numba.jit(nopython=True)
def mssne_lds_similarities_h_fast(arr_den_s_i, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t):
    """
    Computation of the matrix of Ms SNE asymmetric similarities in LDS at some scale, as defined in [5], as well as their log.
    In the documentation, we denote by
    - dsm_lds: two-dimensional numpy array with shape (N, N), where N is the number of data points. Element [i,j] must be the squared LDS distance between i and j. The diagonal elements are assumed to be equal to np.inf.
    - dsm_lds_min_row: equal to dsm_lds.min(axis=1).
    - N: number of data points.
    In:
    - arr_den_s_i: numpy one-dimensional array with N elements, where element i is the denominator of the exponentials of the p_{ij}'s for some i and for j=0, ..., N-1. Hence element i is equal to 2*(sigma_i**2). It is assumed that arr_den_s_i == np.maximum(np_eps, arr_den_s_i).
    - np_eps: equal to np.finfo(dtype=np.float64).eps.
    - arr_ones: equal to np.ones(shape=N, dtype=np.float64).
    - dsm_lds_min_row_dsm_lds_t: equal to dsm_lds_min_row-dsm_lds.T.
    Out:
    A two-dimensional numpy array with shape (N, N) and in which element [i,j] = q_{ij}. q_{ii} is set to 0.
    """
    # Numerators of the similarities
    q_ij = np.exp(dsm_lds_min_row_dsm_lds_t/arr_den_s_i)
    # Correcting the diagonal of the similarities. Using a loop for numba
    q_ij = fill_diago(M=q_ij, v=0.0).astype(np.float64)
    # Computing the N**2 LDS similarities q_{ij}, for i, j = 0, ..., N-1, and returning.
    return (q_ij/np.maximum(np_eps, np.dot(arr_ones.astype(np.float64), q_ij))).T


def mssne_obj_fct(x, sigma_ij, N, n_components, p_h, t_h, n_perp):
    """
    Computes the value of the objective function of Multi-scale SNE.
    In:
    - x: numpy one-dimensional array with N*n_components elements, containing the current values of the low dimensional coordinates. np.reshape(a=x, newshape=(N, n_components)) should yield a two-dimensional array with one example per row and one LDS coordinate per column.
    - sigma_ij: numpy two-dimensional array with shape (N,N). Element (i,j) should contain sigma_{ij}, as defined in [10]. Diagonal elements must be equal to 0.
    - N: number of data points.
    - n_components: dimension of the LDS.
    - n_perp: number of perplexities which are considered.
    - p_h: one-dimensional numpy array containing n_perp elements and in which element h contains the LDS precision associated with the h^th considered perplexity.
    - t_h: one-dimensional numpy array containing n_perp elements and in which element h contains the LDS bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h == np.maximum(np_eps, t_h).
    Out:
    A scalar representing the Multi-scale SNE objective function evaluation.
    """
    global n_eps_np_float64
    # Evaluating the LDS similarities.
    s_ij = mssne_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h, np_eps=n_eps_np_float64, arr_ones=np.ones(shape=N, dtype=np.float64), dsm_lds_min_row_dsm_lds_t=mssne_eval_dsm_lds_min_row_dsm_lds_t(X=np.reshape(a=x, newshape=(N, n_components))))[0]
    # Computing the cost function value
    return scipy.special.rel_entr(sigma_ij, s_ij).sum()

@numba.jit(nopython=True)
def mssne_eval_grad(N, n_perp, t_h, np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t, n_components, p_h, X, sigma_ij):
    """
    Evaluate the Ms SNE gradient.
    In:
    - N: number of data points.
    - n_perp: number of perplexities which are considered.
    - t_h: one-dimensional numpy array containing n_perp elements and in which element h contains the LDS bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h == np.maximum(np_eps, t_h).
    - np_eps, arr_ones, dsm_lds_min_row_dsm_lds_t: same as in function mssne_lds_similarities_h_fast.
    - n_components: dimension of the LDS.
    - p_h: one-dimensional numpy array containing n_perp elements and in which element h contains the LDS precision associated with the h^th considered perplexity.
    - X: numpy two-dimensional array with shape (N, n_components), containing the current values of the low dimensional coordinates. It contains one example per row and one LDS coordinate per column.
    - sigma_ij: numpy three-dimensional array with shape (n_samp, N, N). For each k in range(n_samp), sigma_ij[k,i,j] should contain sigma_{ij}, as defined in [10] for the k^th data set. Elements for which i=j must be equal to 0.
    Out:
    A two-dimensional numpy array with shape (N, n_components) storing the evaluation of the Ms SNE gradient.
    """
    # Computing the LDS similarities
    s_ij, s_hij = mssne_eval_sim_lds(N=N, n_perp=n_perp, t_h=t_h, np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_lds_min_row_dsm_lds_t)
    # Computing the quotient of sigma_ij by s_ij
    ss_ij = sigma_ij/s_ij
    # Computing the gradient
    grad = np.zeros(shape=(N, n_components), dtype=np.float64)
    for h in range(n_perp):
        # Computing the product between ss_ij and s_hij[h,:,:], summing over the columns, substracting the result from each row of ss_ij and multiplying by s_hij[h,:,:].
        Mh = s_hij[h,:,:]*((ss_ij.T - np.dot(ss_ij*s_hij[h,:,:], arr_ones)).T)
        Mh += Mh.T
        # Updating the gradient
        grad += p_h[h]*((X.T*np.dot(Mh, arr_ones)).T - np.dot(Mh, X))
    # Returning
    return grad/np.float64(n_perp)

def mssne_grad(x, sigma_ij, N, n_components, p_h, t_h, n_perp):
    """
    Computes the value of the gradient of the objective function of Multi-scale SNE.
    In:
    - x: numpy one-dimensional array with N*n_components elements, containing the current values of the low dimensional coordinates. np.reshape(a=x, newshape=(N, n_components)) should yield a two-dimensional array with one example per row and one LDS coordinate per column.
    - sigma_ij: numpy two-dimensional array with shape (N,N). Element (i,j) should contain sigma_{ij}, as defined in [10]. Diagonal elements must be equal to 0.
    - N: number of data points.
    - n_components: dimension of the LDS.
    - n_perp: number of perplexities which are considered.
    - p_h: one-dimensional numpy array containing n_perp elements and in which element h contains the LDS precision associated with the h^th considered perplexity.
    - t_h: one-dimensional numpy array containing n_perp elements and in which element h contains the LDS bandwidth associated with the h^th considered perplexity, which is equal to 2.0/p_h[h]. It is assumed that t_h == np.maximum(np_eps, t_h).
    Out:
    A one-dimensional numpy array with N*n_components elements, where element i is the coordinate of the gradient associate to x[i].
    """
    global n_eps_np_float64
    X = np.reshape(a=x, newshape=(N, n_components))
    # Evaluating the gradient
    grad = mssne_eval_grad(N=N, n_perp=n_perp, t_h=t_h, np_eps=n_eps_np_float64, arr_ones=np.ones(shape=N, dtype=np.float64), dsm_lds_min_row_dsm_lds_t=mssne_eval_dsm_lds_min_row_dsm_lds_t(X=X), n_components=n_components, p_h=p_h, X=X, sigma_ij=sigma_ij)
    # Returning the reshaped gradient
    return np.reshape(a=grad, newshape=N*n_components)

def mssne_sim_hds_bandwidth(X_hds, K_h, N, n_components, X_lds, fit_U=True, dsm_hds=None):
    """
    Evaluates the MsSNE multi-scale HDS bandwidths and the LDS bandwidths and precisions, or directly returns the HDS similarities at the different scales.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the high dimensional data set, with one row per example and one column per dimension.
    - K_h: one-dimensional numpy array, with the perplexities at each scale in increasing order.
    - N: number of data points.
    - n_components: numbr of components of the LDS.
    - X_lds: two-dimensional numpy array with shape (N, n_components) containing the current value of the low-dimensional embedding.
    - fit_U: same as for msjse_lds_bandwidths.
    - dsm_hds: (optional) two-dimensional numpy array with the pairwise SQUARED HDS distances between the data points. If None, deduced from X_hds using squared euclidean distances.
    Out:
    If fit_U is True, a tuple with:
    - dsm_hds: two-dimensional numpy array with the SQUARED pairwise HDS distances between the data points.
    - tau_hi: a two-dimensional numpy array in which element tau_hi[h,i] contains tau_{hi} = 2/pi_{hi}, as defined in [10].
    - p_h: one-dimensional numpy array with the LDS precisions at each scale defined by K_h.
    - t_h: one-dimensional numpy array with the LDS bandwidths at each scale defined by K_h.
    If fit_U is False, we return a three-dimensional numpy array with shape (K_h.size, N, N) where sim_hij[h,:,:] contains the HDS similarities at scale K_h[h].
    """
    # Computing the HDS distances. We use "sqeuclidean" metric instead of "euclidean" to avoid squaring the distances.
    if dsm_hds is None:
        dsm_hds = pairwise_dist(X=X_hds, metric='sqeuclidean', tomatrix=True)
    if fit_U:
        # Computing the multi-scale HDS bandwidths and the LDS bandwidths and precisions
        tau_hi, D_h, U, p_h, t_h = msjse_hlds_bandwidths(dsm_hds=dsm_hds, K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U)
        # Returning
        return dsm_hds, tau_hi, p_h, t_h
    else:
        # Returning the HDS similarities at the scales indicated by K_h
        return msjse_hds_similarities(dsm_hds=dsm_hds, arr_perp=K_h)[2]

def mssne_implem(X_hds, init, n_components=2, ret_sim_hds=False, fit_U=True, dm_hds=None, cur_seed=None):
    """
    This function applies Multi-scale SNE to reduce the dimensionality of a data set to n_components dimensions.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the high dimensional data set, with one row per example and one column per dimension.
    - init: see function sne_init_embedding for a description.
    - n_components: dimension of the LDS.
    - ret_sim_hds: boolean. If true, the multi-scale HDS similarities are also returned.
    - fit_U: boolean indicating whether to fit the U in the definition of the LDS similarities. If True, the U is tuned as in [10]. Otherwise, it is forced to 1.
    - dm_hds: (optional) two-dimensional numpy array with the pairwise HDS distances (NOT squared) between the data points. If None, deduced from X_hds using euclidean distances. If not None, then X_hds must be None and init must be 'random', otherwise an error is produced.
    - cur_seed: seed to use for the random state. If None, 0 is used. 
    Out:
    If ret_sim_hds is False, a numpy.ndarray X_lds with shape (N, n_components), containing the low dimensional data set, with one row per example and one column per dimension.
    If ret_sim_hds is True, a tuple with:
    - X_lds as described in the case of ret_sim_hds = False.
    - a two-dimensional numpy array with shape (N, N), where N is the number of samples. It contains the multi-scale pairwise HDS similarities between the samples. 
    Remarks:
    - L-BFGS algorithm is used, as suggested in [10].
    - Multi-scale optimization is performed, as presented in [10].
    """
    global sim_dr_nitmax, sim_dr_gtol, sim_dr_ftol, sim_dr_maxls, sim_dr_maxcor, n_eps_np_float64, module_name
    # Checking the value of dm_hds
    dm_hds_none = dm_hds is None
    if dm_hds_none:
        dsm_hds = None
    else:
        if (X_hds is not None) or (init != 'random'):
            raise ValueError("Error in function mssne_implem of module {module_name}: if dm_hds is not None, then X_hds must be None and init must be 'random'.".format(module_name=module_name))
        dsm_hds = dm_hds**2
        dsm_hds = dsm_hds.astype(np.float64)
    # Defining the random state
    if cur_seed is None:
        cur_seed = 0
    rand_state = np.random.RandomState(cur_seed)
    # Number of data points
    if dm_hds_none:
        N = X_hds.shape[0]
    else:
        N = dsm_hds.shape[0]
    if fit_U:
        arr_ones = np.ones(shape=N, dtype=np.int64)
    # Product of N and n_components
    prod_N_nc = N*n_components
    # Maximum number of L-BFGS steps at each stage of the multi-scale optimization.
    nit_max = sim_dr_nitmax
    # Tolerance for the norm of the gradient in the L-BFGS algorithm
    gtol = sim_dr_gtol
    # Tolerance for the relative update of the value of the objective function.
    ftol = sim_dr_ftol
    # Smallest float
    np_eps = n_eps_np_float64
    # Function to compute the gradient of the objective.
    fct_grad = mssne_grad
    # Function to compute the objective function
    fct_obj = mssne_obj_fct
    
    ## Parameters used for the L-BFGS optimization
    # Maximum number of line search steps (per L-BFGS iteration).
    maxls = sim_dr_maxls
    # The maximum number of variable metric corrections used to define the limited memory matrix. 
    maxcor = sim_dr_maxcor
    
    # Defining K_star for the multi-scale perplexities
    K_star = 2
    # Computing the multi-scale perplexities
    L_min, L_max, L, K_h = ms_perplexities(N=N, K_star=K_star)
    
    # Initializing the low dimensional embedding.
    if dm_hds_none:
        X_lds = sne_init_embedding(X_hds=X_hds, init=init, n_components=n_components, rand_state=rand_state)
    else:
        X_lds = sne_init_random_embedding(N=N, n_components=n_components, rand_state=rand_state)
    
    # Computing the multi-scale HDS bandwidths if fit_U is True, and the HDS similarities at the different scales otherwise. We also evaluate the LDS bandwidths and precisions.
    retv_mssne_sim = mssne_sim_hds_bandwidth(X_hds=X_hds, K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U, dsm_hds=dsm_hds)
    if fit_U:
        dsm_hds, tau_hi, p_h, t_h = retv_mssne_sim
        dsm_hds_min_row_dsm_lds_t = mssne_eval_dsm_lds_min_row_dsm_lds_t_dsm(dsm_lds=dsm_hds)
    else:
        sim_hij_allh = retv_mssne_sim
        p_h, t_h = msjse_lds_bandwidths(tau_hi=np.empty(shape=(L, N), dtype=np.float64), K_h=K_h, N=N, n_components=n_components, X_lds=X_lds, fit_U=fit_U)[2:]
    
    # Reshaping X_lds as the optimization functions only work with one-dimensional arrays.
    X_lds = np.reshape(a=X_lds, newshape=prod_N_nc)
    
    # Matrix storing the multi-scale HDS similarities sigma_{ij}. Element [i,j] contains sigma_{ij}. sigma_{ii} is set to 0. 
    sigma_ij = np.zeros(shape=(N,N), dtype=np.float64)
    
    # Multi-scale optimization. n_perp is the number of currently considered perplexities.
    for n_perp in range(1, L+1, 1):
        # Index of the currently added perplexity.
        h = L-n_perp
        # Computing the LDS precisions
        cur_p_h = p_h[h:]
        # Computing the LDS bandwidths
        cur_t_h = t_h[h:]
        
        # Computing the N**2 HDS similarities sigma_{hij} if fit_U is True, using the bandwidths tau_hi which were already computed. Otherwise, we just gather them.
        if fit_U:
            sigma_hij = mssne_lds_similarities_h_fast(arr_den_s_i=tau_hi[h,:], np_eps=np_eps, arr_ones=arr_ones, dsm_lds_min_row_dsm_lds_t=dsm_hds_min_row_dsm_lds_t)
        else:
            sigma_hij = sim_hij_allh[h,:,:]
        # Updating the multi-scale HDS similarities
        sigma_ij = (sigma_ij*(np.float64(n_perp)-1.0) + sigma_hij)/np.float64(n_perp)
        
        # Defining the arguments of the L-BFGS algorithm
        args = (sigma_ij, N, n_components, cur_p_h, cur_t_h, n_perp)
        
        # Running L-BFGS
        res = lbfgsb(fun=fct_obj, x0=X_lds, args=args, jac=fct_grad, gtol=gtol, ftol=ftol, maxiter=nit_max, maxls=maxls, maxcor=maxcor, disp=False)
        X_lds = res.x
    
    # Reshaping the result
    X_lds = np.reshape(a=X_lds, newshape=(N, n_components))
    
    # Returning
    if ret_sim_hds:
        return X_lds, sigma_ij
    else:
        return X_lds


##############################
############################## Basic tests. 
##############################


def plot_leo(X_lds, labels, tit, labs=0, classes_names=0):
    
    N= len(set(labels))			
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		
    a1 = plt.scatter(X_lds[:, 0], X_lds[:, 1],  c=labels, cmap=cmap, s=20, norm=norm)
    
    if labs==1:    
       for i in range(X_lds.shape[0]):              
           plt.text(X_lds[i, 0], X_lds[i, 1], str(i), fontdict={'size': 7})

    plt.xlim([2*X_lds.min(),2*X_lds.max()])
    plt.ylim([2*X_lds.min(),2*X_lds.max()])
    plt.title(tit)

    lp = lambda i: plt.plot([],color=a1.cmap(a1.norm(i)), ms=10.0, mec="none", ls="", marker="o")[0]                
    handles = [lp(i) for i in list(set(labels))]
    plt.legend(labels=classes_names, loc=2)

    plt.show()	


###
### Multi-scale SNE (convenient as no perplexity is needed => valid for all data set sizes). If you provide X_hds, then Euclidean distance are used in the HD space. If you want to use another HD distance, provide in dm_hds a two-dimensional numpy array with the pairwise HD distances and set init and X_hds to None. 
###


init = 'random'
print("Applying Ms SNE")

#X_lds = mssne_implem(X_hds=X_hds, init=init, n_components=2, dm_hds=None); print("Without similarity matrix");
###X_lds = mssne_implem(X_hds=None, init=init, n_components=2, dm_hds=X_hds); print("With similarity matrix");

#plot_leo(X_lds, y, 'Multi-scale SNE: auc={v}'.format(v=1), labs=0,classes_names=classes_names)

