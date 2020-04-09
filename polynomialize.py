import numpy as np

def num_of_terms(n):
    if int(n) != n:
        raise TypeError('n must be integer')
    # Can be found by solving the recurrence relation
    # R(n) = R(n - 1) + (n + 1), R(1) = 3
    return int (  (n+4) * (n-1) / 2 + 3  )

def calc_poly_term(x1, x2, i, j):
    return ( x1 ** (i-j) ) * ( x2 ** j )

def polynomialize(x1, x2, n, coef=None):
    """
    
        Function that returns all of the n-th degree polynomial
        terms of two given data

        Example: x1 = [1, 2, 3], x2 = [1, 2, 3], n = 2

        Terms of a 2nd degree polynomial function: 1, x, y,  x^2, xy, y^2

        polynomialize(np.array([1, 2, 3]), np.array([1, 2, 3]), 2) will return:

                1 |  x |  y | x^2 | xy | y^2
        
        array([ [1., 1., 1., 1., 1., 1.],
                [1., 2., 2., 4., 4., 4.],
                [1., 3., 3., 9., 9., 9.]])
        where each column is the evaluated polynomial terms given above, respectively.

        If you also give a `coef` parameter, which must be a vector of size of the number of 
        terms, elements of coef will be multiplied by the corresponding column.

        Example: polynomialize(np.array([1, 2, 3]), np.array([1, 2, 3]), 2, [1, 0, 0, 1, 0, 0]) will return:

        array([ [1., 0., 0., 1., 0., 0.],
                [1., 0., 0., 4., 0., 0.],
                [1., 0., 0., 9., 0., 0.]])
        multiplying the first and fourth column with 1, and the other columns with 0. 

    """


    num_terms = num_of_terms(n)
    
    if coef is None:
        coef = np.ones(num_terms)
    elif len(coef) != num_terms:
        raise TypeError('Coef must have the same size of num_features')
     
    if x1.shape != x2.shape:
        raise TypeError('x1 and x2 must be of the same shape!')
    
    
    k_term = 0

    result = np.ones(x1.shape + (num_terms,))

    for i in range(0, n + 1):
        for j in range(0, i + 1):
            value = calc_poly_term(x1, x2, i, j)
            result[..., k_term] = value * coef[k_term]
            k_term += 1
        
    return result
