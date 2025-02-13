from imblearn.under_sampling import NearMiss
import numpy as np
import sys 
import qiskit.quantum_info as qi

def normalize_statevector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

X=np.loadtxt(sys.argv[1])
y=np.loadtxt(sys.argv[2])
print('Samples in class 0:', sum(y == 0))
print('Samples in class 1:', sum(y == 1))

nm = NearMiss()


# Fit predictor (X) and target (y) using fit_resample()
X_nearmiss, y_nearmiss = nm.fit_resample(X, y)
X_final_DM=[]
print(len(X_nearmiss))
print('Samples in class 0:', sum(y_nearmiss == 0))
print('Samples in class 1:', sum(y_nearmiss == 1))

for x in X_nearmiss:
    x_normalized=normalize_statevector(x)
    x_DM=qi.DensityMatrix(x_normalized)
    assert(x_DM.is_valid())
    X_final_DM.append(x_DM.data.flatten())

np.savetxt(sys.argv[3],X_final_DM)
np.savetxt(sys.argv[4],y_nearmiss)