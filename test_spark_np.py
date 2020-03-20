import numpy as np
 
from pyspark import SparkContext
from pyspark import SparkConf
 
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



