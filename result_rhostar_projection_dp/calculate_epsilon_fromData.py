import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

delta=0.05

data=np.loadtxt(sys.argv[1])
num_data=len(data[0])
num_bins=int(num_data/20)
print(int(num_bins))
histo,edge=np.histogram(data[0], bins=num_bins, density=True)
histo_adj,edge_adj=np.histogram(data[1], bins=num_bins, density=True)
        


pdf = histo
pdf_adj=histo_adj
pdf_P = np.array(pdf)
pdf_Q = np.array(pdf_adj)

plt.plot(pdf_P,pdf_Q)

# Avoid division by zero or log of negative numbers
valid = (pdf_Q > 0) & (pdf_P > delta)

ratio = (pdf_P[valid] - delta) / pdf_Q[valid]
plt.show()
# Compute the supremum
epsilon = np.max(np.log(ratio))
print('worst case',epsilon)



