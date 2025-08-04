import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

delta=0.01

data=np.loadtxt(sys.argv[1])
num_data=len(data[0])
#num_bins=int(num_data/20)
print(int(num_data))

binss=np.arange(0,1,0.01)
histo,edge=np.histogram(data[0], bins=binss, density=True)
histo_adj,edge_adj=np.histogram(data[1], bins=binss, density=True)

assert(edge.all()==edge_adj.all())
pdf = histo
pdf_adj=histo_adj
pdf_P = np.array(pdf)
pdf_Q = np.array(pdf_adj)

# Avoid division by zero or log of negative numbers
valid = (pdf_Q > 0) & (pdf_P > delta)
print(len(pdf_P[valid]))
ratio = (pdf_P[valid] - delta) / pdf_Q[valid]

plt.plot(edge[1:],pdf)
plt.plot(edge_adj[1:],pdf_adj)
print(pdf_P[valid])
print(pdf_Q[valid])
#plt.hist(ratio)
plt.show()
# Compute the supremum
epsilon = np.max(np.log(ratio))
print('worst case',epsilon)



