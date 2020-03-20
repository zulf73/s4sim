from numba import jit
from math import sin, cos, log
import math
#from sympy import *
import numpy as np
from symengine import var


def norm(v):
    t = 0
    for ve in v:
        t = t + ve**2
    return np.sqrt(t)


multi_indices_cache = np.array([[0], ])


def multi_indices(k):
    k = int(k)
    global multi_indices_cache
    if k == 0:
        multi_indices_cache = np.array([np.array([0])])
    if len(multi_indices_cache) > k:
        return multi_indices_cache[k]
    multi_indices(k-1)
    ans = []
    for s1 in range(k + 5 - 1 - 3):
        for s2 in range(s1+1, k + 5 - 1 - 2):
            for s3 in range(s2+1, k + 5 - 1 - 1):
                for s4 in range(s3+1, k + 5 - 1):
                    a = [0]*5
                    a[0] = s1
                    a[1] = s2 - s1 - 1
                    a[2] = s3 - s2 - 1
                    a[3] = s4 - s3 - 1
                    a[4] = k - sum(a[:4])
                    # print(a)
                    ans.append(a)
    np.append(multi_indices_cache, [ans])
    return(np.array(ans))


# hack to make sure the multi-indices are set up
for k in range(1, 23):
    cA = multi_indices(k)


# @jit(nopython=True)
def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def clean_coeff_dict(p):
    pt = p.expand()
    # print(pt)
    #pt = pt.subs((x**2)**0.5, x)
    #pt = pt.subs((y**2)**0.5, y)
    #pt = pt.subs((z**2)**0.5, z)
    #pt = pt.subs((u**2)**0.5, u)
    #pt = pt.subs((v**2)**0.5, v)
    pdict = pt.as_coefficients_dict()
    # print(len(pdict.keys()))
    return pdict


A1 = multi_indices(1)


def monomials_equal(m1, m2):
    # evalate them
    a = np.array([1, 1.2, 1.4, 1.6, 1.8])
    v1 = m1.subs([x, y, z, u, v], [a[0], a[1], a[2], a[3], a[4]])
    v2 = m2.subs([x, y, z, u, v], [a[0], a[1], a[2], a[3], a[4]])
    if abs(float(v1)-float(v2)) > 1e-5:
        return False
    else:
        return True

    return True


def coef(expr, alpha):
    x, y, z, u, v = var('x y z u v')
    e1 = expr.coeff(x, alpha[0])
    e2 = e1.coeff(y, alpha[1])
    e3 = e2.coeff(z, alpha[2])
    e4 = e3.coeff(u, alpha[3])
    e5 = e4.coeff(v, alpha[4])
    return(e5)


def coef_from_dict(cfdict, alpha):
    monomial = x**alpha[0] * y**alpha[1] * \
        z**alpha[2] * u**alpha[3] * v**alpha[4]
    for mon in cfdict.keys():
        if monomials_equal(mon, monomial):
            return cfdict[mon]
    return(0.0)

# @jit(nopython=True)


def multi_index_of_monomial(mon):
    ans = [0.]*5
    vr = [1.]*5
    for rr in range(5):
        vr[rr] = 2.0
        zr = mon.subs([x, y, z, u, v], vr)
        if abs(float(zr)) > 0:
            zz = log(abs(float(zr)))/log(2.0)
            ans[rr] = int(zz)
    return ans


def dim_hom_poly_fixed_degree(max_degree):
    k = max_degree
    dimHom = dimHom + choose(k+5-1, k)
    return(dimHom)


def dim_hom_poly(max_degree):
    dimHom = 0
    for k in range(max_degree):
        dimHom = dimHom + choose(k+5-1, k)

    return(dimHom)


x, y, z, u, v = var('x y z u v')


def index_of_mon_in_homogeneous(midx):
    k = sum(midx)
    A = multi_indices(k)
    for j in range(len(A)):
        if np.linalg.norm(A[j]-midx) < 1e-5:
            return(j)
    return(0)


def coeffs_in_hom_poly_basis(p, max_degree):

    #print('in coeffs_in_hom_poly_basis')
    # print('polynomial')
    # print(p)

    cfdict = clean_coeff_dict(p)
    mons = cfdict.keys()
    dimHom = dim_hom_poly(max_degree)
    vec = [0.] * dimHom
    for mon in mons:
        midx = multi_index_of_monomial(mon)
        degree = sum(midx)
        pdhp = dim_hom_poly(degree - 1)
        if degree > max_degree:
            next
        idx = pdhp + index_of_mon_in_homogeneous(midx)
        if idx < dimHom:
            vec[idx] = float(cfdict[mon])

    return(vec)


def coeffs_in_fixed_degree_basis(p, max_degree):

    #print('in coeffs_in_hom_poly_basis')
    # print('polynomial')
    # print(p)

    cfdict = clean_coeff_dict(p)
    mons = cfdict.keys()
    dimHomFixed = dim_hom_poly_fixed_degree(max_degree)
    vec = [0.] * dimHomFixed
    for mon in mons:
        midx = multi_index_of_monomial(mon)
        degree = sum(midx)
        if degree > max_degree:
            next
        idx = index_of_mon_in_homogeneous(midx)
        if idx < dimHomFixed:
            vec[idx] = float(cfdict[mon])
    return(vec)


if False:
    """
p = (1 + 0.5*(u + v + x + y + z) + 0.333333333333333*(u + v + x + y + z)**2 + 0.25*(u + v + x + y + z)**3 + 0.2*(u + v + x + y + z)**4
     + 0.166666666666667*(u + v + x + y + z)**5)*(-2.98242314016196*(v**2)**(-2.5)*((v**2)**0.5)**7.0 + 14.91211570081*v**2*(v**2)**(-3.5)*((v**2)**0.5)**7.0)
+ 16*((u**2 + v**2 + x**2 + y**2 + z**2)**0.5)**7.0*(15.0*v**2*(u**2 + v**2 +
                                                                x**2 + y**2 + z**2)**(-3.5) - 3.0*(u**2 + v**2 + x**2 + y**2 + z**2)**(-2.5))

cfdict = p.expand().as_coefficients_dict()


pv = coeffs_in_hom_poly_basis(p, 10)
"""


# @jit(nopython=True)


def coef_vector(expr, k):
    A = multi_indices(k)
    cfdict = clean_coeff_dict(expr)
    vv = []
    for a in A:
        cf = coef_from_dict(cfdict, a)
        vv.append(cf)
    return vv


def rotate(h, p1, p2, t):
    if p2 < p1:
        temp = p1
        p1 = p2
        p2 = temp
    g = h
    g2 = None
    ct = cos(t)
    st = sin(t)
    # print '%d,%d' % (p1, p2)
    # print '%.3f,%.3f' % (ct, st)
    x, y, z, u, v = var('x y z u v')
    if p1 == 1:
        if p2 == 2:
            g2 = g.subs([(x, ct*x + st*y), (y, -st*x + ct*y)])
        if p2 == 3:
            g2 = g.subs([(x, ct*x + st*z), (z, -st*x + ct*z)])
        if p2 == 4:
            g2 = g.subs([(x, ct*x + st*u), (u, -st*x + ct*u)])
        if p2 == 5:
            g2 = g.subs([(x, ct*x + st*v), (v, -st*x + ct*v)])
    if p1 == 2:
        if p2 == 3:
            g2 = g.subs([(y, ct*y + st*z), (z, -st*y + ct*z)])
        if p2 == 4:
            g2 = g.subs([(y, ct*y + st*u), (u, -st*y + ct*u)])
        if p2 == 5:
            g2 = g.subs([(y, ct*y + st*v), (v, -st*y + ct*v)])
    if p1 == 3:
        if p2 == 4:
            g2 = g.subs([(z, ct*z + st*u), (u, -st*z + ct*u)])
        if p2 == 5:
            g2 = g.subs([(z, ct*z + st*v), (v, -st*z + ct*v)])
    if p1 == 4:
        if p2 == 5:
            g2 = g.subs([(u, ct*u + st*v), (v, -st*u + ct*v)])

    return(g2.expand())


def rotate_by_unit_vector(h, unit_vector, theta):
    g = h
    ct = cos(theta)
    st = sin(theta)
    # print '%d,%d' % (p1, p2)
    # print '%.3f,%.3f' % (ct, st)

    v1 = unit_vector[0]
    v2 = unit_vector[1]
    v3 = unit_vector[2]
    v4 = unit_vector[3]
    x, y, z, u, v = var('x y z u v')
    g2 = g.subs(dict([(x, st*x*v1), (y, st*y*v2),
                      (z, st*z*v3), (u, st*u*v4), (v, ct*v)]))

    return(g2.expand())


# @jit(nopython=True)
def dim_hom_poly(max_degree):
    dimHom = 0
    for k in range(int(max_degree)):
        dimHom = dimHom + choose(k+5-1, k)

    return(dimHom)


# @jit(nopython=True)
def basis_of_homogeneous_polynomials(max_degree):
    dimHom = dim_hom_poly(max_degree)
    return np.eye(dimHom)

# assume two poerations 'poly' and 'pvect' which go between polynomials and their
# representations


def pvect(p, max_degree):
    # get coefficients of p and cut off at
    v = coeffs_in_hom_poly_basis(p,  max_degree)
    return v


def idx2midx_fixed(j, deg):
    # determine total degree
    A = multi_indices(deg)
    return A[j]


# we need idx2midx and monomial
# need to check for accuracy
# @jit(nopython=True)
def idx2midx(j):
    # determine total degree
    start = 0
    k = 0
    while start < j:
        start = start + choose(k+5-1, k)
        k = k+1
    # start = start - choose(k+5-1, k)
    # k = k - 1
    A = multi_indices(k)
    return A[j - start]


# @jit(nopython=True)
def monomial(a):
    x, y, z, u, v = var('x y z u v')
    return x**a[0]*y**a[1]*z**a[2]*u**a[3]*v**a[4]


# @jit(nopython=True)
def poly(vec):
    p = 0
    for j in range(len(vec)):
        midx = idx2midx(j)
        r = monomial(midx)
        p = p + vec[j] * r
    return(p)


def poly_fixed(vec, k):
    p = 0
    for j in range(len(vec)):
        midx = idx2midx_fixed(j, k)
        r = monomial(midx)
        p = p + vec[j] * r
    return(p)


def square_matrix(B):
    nr, nc = B.shape
    diff = nr - nc
    if diff > 0:
        Z = np.zeros((nr, diff))
        Bprime = np.concatenate((B, Z), axis=1)
        return(Bprime)
    if diff < 0:
        Z = np.zeros((-diff, nc))
        Bprime = np.concatenate((B, Z), axis=0)
        return(Bprime)
    return(B)


# hack to make sure the multi-indices are set up
for k in range(1, 23):
    cA = multi_indices(k)


def differentiate(f, a):
    df = f
    for inda in range(5):
        if a[inda] > 0:
            if inda == 0:
                df = df.diff(x, a[inda])
            if inda == 1:
                df = df.diff(y, a[inda])
            if inda == 2:
                df = df.diff(z, a[inda])
            if inda == 3:
                df = df.diff(u, a[inda])
            if inda == 4:
                df = df.diff(v, a[inda])
    return(df)
