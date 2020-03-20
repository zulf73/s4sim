import numpy as np

from pyspark import SparkContext
from pyspark import SparkConf

from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import pandas as pd
from biopandas.pdb import PandasPdb
import configparser
from mpmath import gegenbauer
from symengine import *
from periodictable import *
from s4utils import *


conf = SparkConf()
conf.setMaster("local")
conf.setAppName("NumpyMult")
sc = SparkContext(conf=conf)


def mult(x):
    y = np.array([2])
    return x*y


x = np.arange(10000)
distData = sc.parallelize(x)

out = distData.map(mult)

print(out)

results = out.collect()

print(results)


# Hacks to see if Spark works
spark = SparkSession.builder.master(
    "local").appName("test pandas").getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Generate a Pandas DataFrame
pdf = pd.DataFrame(np.random.rand(100, 3))

# Create a Spark DataFrame from a Pandas DataFrame using Arrow
df = spark.createDataFrame(pdf)

# Convert the Spark DataFrame back to a Pandas DataFrame using Arrow
result_pdf = df.select("*").toPandas()


SparkContext.addPyFile(
    "/mnt/c/Users/zulfi/OneDrive/S4Physics/gromacs/symengine-0.5.1-py3.7.egg")


# Translation between R^3 and S^4


def r3point_to_s4point(x1, x2, x3, a):
    r = np.sqrt(x1**2 + x2**2 + x3**2)
    if r < a:
        x4 = - np.sqrt(a**2 - r**2)
        return([x1, x2, x3, x4])
    else:
        # north pole is (0,0,0,a)
        # find the intersection of S^4(a) and line
        # connecting (x1,x2,x3,0) to the north pole
        # that line is l(t) = t (x1, x2, x3, 0) + (1-t) (0,0,0,a)
        #
        # Therefore at the intersection
        # r^2/a^2 = 2/t - t
        # this gives us the quadratic equation
        # t^2 + (r^2/a^2)*t - 2 = 0
        b = r**2/a**2
        t = - 0.5 * (- b + np.sqrt(b**2 - 8))
        return([t*x1, t*x2, t*x3, (1-t)*a, 0])


def s4point_to_r3point(x1, x2, x3, x4, x5):
    a = np.sqrt(x1**2 + x2**2 + x3**2 + x4**2)
    r = np.sqrt(x1**2 + x2**2 + x3**2)
    if x4 < 0:
        return([x1, x2, x3])
    else:
        t = x4/(x4-a)
        return([(1-t)*x1, (1-t)*x2, (1-t)*x3])

# Need zonal harmonic polynomial associated with a point
# z in S^4


def univariate_gegenbauer(n, alpha, q):
    g = gegenbauer(n, alpha, q)
    return g


def zonal_polynomial(z1, z2, z3, z4, z5, n):
    x, y, z, u, v, q = vars('x y z u v q')
    g = (3/2*n+1) * univariate_gegenbauer(3/2, n, q)
    product_zx = z1*x + z2*y + z3*z + z4*u + z5*v
    h = g.subs(q, product_zx)
    return(h)


def mass_of(element_symbol):
    # send out daltons
    mass = 0
    for el in elements:
        if el.symbol == element_symbol:
            mass = el.mass
    return mass


def create_pdb_from_model_matrix(model_matrix, old_pdb, pdb_name):
    new_coordinates = []
    for column in model_matrix:
        zp = closest_polynomial_center(column)
        np.append(new_coordinates, s4point_to_r3point(zp))
    y = old_pdb
    atoms = y.df['ATOM']
    atoms[['x_coord', 'y_coord', 'z_coord']] = new_coordinates
    y.df['ATOM'] = atoms
    y.to_pdb(path='./'+pdb_name, records=None, gz=False, spprnd_newline=True)


# Read in an important ptotein file 1PSI, 1K90, ZLQU
x = PandasPdb().fetch_pdb('1kqu')

r3points = x.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]

# initialize the matrix on the S4 side
# Now we have an initializzed matrix

N = len(r3points) + 1
nk = dim_hom_poly(8)
model_matrix = np.zeros((nk, N))
a = 1e-6


def initialize_model():
    global model_matrix
    for idx in range(N):
        pt = r3points[idx]
        s4pt = r3point_to_s4point(pt[0], pt[1], pt[2], a)

        for deg in range(8):
            zonal = zonal_polynomial(
                s4pt[0], s4pt[1], s4pt[2], s4pt[3], s4pt[4], deg)

        # put down the coordinates for the column
        model_matrix[:, idx] = zonal_coefficients(zonal)
        asquared = 0.0
        for k in range(nk):
            asquared = sum(model_matrix[k, :(N-1)]**2)
            model_matrix[k, -1] = np.sqrt(asquared)


def update_model(time, prev_time):
    dt = time - prev_time
    els = x.df['ATOM']['element_symbol']
    for bn in range(N):
        hbar = 1
        mass = 1.
        try:
            mass = mass_of(els[bn])
        except:
            pass
        c = 1e8
        ev = 0
        for s in range(nk):
            kappa = (mass * c**2 - ev)/hbar
            coupling_const = 1/137.
            alpha = 0
            for jj in range(s):
                jjp = s - jj
                a = model_matrix[jjp, bn]
                b = model_matrix[jj, bn]
                alpha = alpha + a*b
            alpha = coupling_const * alpha
            b = model_matrix[s, bn]
            update_bn = dt * (kappa * b + alpha)
            model_matrix[s, bn] = model_matrix[s, bn] + update_bn


def run_model():
    times = np.arange(0, 0.3, 0.1)
    for j in range(1, len(times)):
        ptime = times[j-1]
        ctime = times[j]
        update_model(ptime, ctime)
        pdb_name = "temp%d.pdb" % j
        create_pdb_from_model_matrix(model_matrix, x, pdb_name)


class MModel:
    def __init__(self):
        config = configparser.ConfigParser()
        master_url = "local"
        app_name = "S4MolecularModel"
        conf = SparkConf().setAppName(app_name) \
                          .setMaster(master_url)
        self.sc = SparkContext(conf=conf)
        self.spark = SparkSession.builder \
                                 .config(conf=conf) \
                                 .getOrCreate()

    def initialize_model(pdbname):
        # Read in an important ptotein file 1PSI, 1K90, ZLQU
        self.ppdb = PandasPdb().fetch_pdb('1kqu')
        r3points = self.ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]

        # initialize the matrix on the S4 side
        # Now we have an initializzed matrix

        N = len(r3points) + 1
        nk = dim_hom_poly(8)
        MM = np.zeros((nk, N))
        a = 1e-6
        for idx in range(N):
            pt = r3points[idx]
            s4pt = r3point_to_s4point(pt[0], pt[1], pt[2], a)

        for deg in range(8):
            zonal = zonal_polynomial(
                s4pt[0], s4pt[1], s4pt[2], s4pt[3], s4pt[4], deg)

            # put down the coordinates for the column
            MM[:, idx] = zonal_coefficients(zonal)
            asquared = 0.0
            for k in range(nk):
                asquared = sum(model_matrix[k, :(N-1)]**2)
                MM[k, -1] = np.sqrt(asquared)

        # create rdd from this
        self.model_matrix = self.sc.parallelize(MM)

    def update_model(time, prev_time):
        dt = time - prev_time
        els = self.ppdb.df['ATOM']['element_symbol']
        for bn in range(N):
            hbar = 1
            mass = 1.
            try:
                mass = mass_of(els[bn])
            except:
                pass
            c = 1e8
            ev = 0
            for s in range(nk):
                kappa = (mass * c**2 - ev)/hbar
                coupling_const = 1/137.
                alpha = 0
                for jj in range(s):
                    jjp = s - jj
                    a = self.model_matrix[jjp, bn]
                    b = self.model_matrix[jj, bn]
                    alpha = alpha + a*b
                    alpha = coupling_const * alpha
                    b = self.model_matrix[s, bn]
                    update_bn = dt * (kappa * b + alpha)
                    self.model_matrix[s,
                                      bn] = self.model_matrix[s, bn] + update_bn

    def run_model():
        times = np.arange(0, 0.3, 0.1)
        for j in range(1, len(times)):
            ptime = times[j-1]
            ctime = times[j]
            self.update_model(ptime, ctime)
            pdb_name = "temp%d.pdb" % j
            create_pdb_from_model_matrix(
                self.model_matrix, self.ppdb, pdb_name)
