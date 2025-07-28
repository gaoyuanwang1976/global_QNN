import numpy as np
from qiskit.circuit import ParameterVector,QuantumCircuit,random
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.quantum_info import SparsePauliOp,Operator
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.circuit.library import RealAmplitudes,ZZFeatureMap,RXGate,EfficientSU2
from qiskit_machine_learning.circuit.library import QNNCircuit
import qiskit.quantum_info as qi
import preprocessing
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 42
import test_core
import copy
import argparse
import global_core
import ray
import math
#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def sigmoid(z,k=10):
    return 1/(1 + np.exp(-k*z))


n_dims=8
n_qubits=int(math.log2(n_dims))
observables = Operator(SparsePauliOp.from_sparse_list(
                [("Z" * n_qubits, range(n_qubits), 1)],
                num_qubits=n_qubits,
            ))

plot_list=[]
num_data=10000

batch_size=[1,2,4,6,8,10,15,20,30,40,50,70,100,150,200]
for batch in batch_size:
    global_list=[]
    instance_list=[]
    glob_positive=0
    instance_positive=0
    for replica in range(num_data):

        circuit_1=random.random_circuit(num_qubits=n_qubits,depth=5)
        circuit_2=random.random_circuit(num_qubits=n_qubits,depth=5)

    
        global_dp_input=np.zeros((n_dims,n_dims),dtype=np.complex128)
        global_dp_output=np.zeros((n_dims,n_dims),dtype=np.complex128)
        instance_difference_total=0
        total_input_expectation=0

        for batch_index in range(batch):
            y_u=-1
            in_data=qi.random_density_matrix(dims=n_dims)

            new_data=in_data.evolve(circuit_1)
            output_data=in_data.evolve(circuit_2)

            input_expectation=new_data.expectation_value(observables)
            output_expectation=output_data.expectation_value(observables)
            instance_difference=(output_expectation-y_u)*(output_expectation-y_u)-(input_expectation-y_u)*(input_expectation-y_u)
            instance_difference_total+=instance_difference

            global_dp_input+=new_data.data
            global_dp_output+=output_data.data

            total_input_expectation+=input_expectation

        input_global_expect=qi.DensityMatrix(global_dp_input/batch).expectation_value(observables)
        
        assert(np.round(np.real_if_close(input_global_expect),3)==np.round(np.real_if_close(total_input_expectation/batch),3))
        output_global_expect=qi.DensityMatrix(global_dp_output/batch).expectation_value(observables)

        global_difference=(output_global_expect-y_u)*(output_global_expect-y_u)-(input_global_expect-y_u)*(input_global_expect-y_u)
        instance_difference_total=instance_difference_total/batch
        global_list.append(global_difference)
        instance_list.append(instance_difference_total)

        if global_difference >0:
            glob_positive+=1
            if instance_difference_total>0:
                instance_positive+=1
    global_list=np.real_if_close(global_list)
    instance_list=np.real_if_close(instance_list)
    
    pearson_corr_sign, p_value_sign = pearsonr(np.sign(global_list), np.sign(instance_list))
    same_sign=sum(np.sign(global_list)== np.sign(instance_list))/len(np.sign(global_list))

    pearson_corr, p_value = pearsonr(global_list, instance_list)
    conditioned_positive=instance_positive/glob_positive
    plot_list.append([pearson_corr,pearson_corr_sign,same_sign])

    print(batch,pearson_corr,pearson_corr_sign,conditioned_positive,same_sign)

plot_list=np.array(plot_list).T
plt.plot(batch_size,plot_list[0],label='PCC',marker='x',markersize=4)
plt.plot(batch_size,plot_list[1],label='sign PCC',marker='>',markersize=4)
plt.plot(batch_size,plot_list[2],label='same sign %',marker='o',markersize=4)
plt.xscale('log')
plt.xlabel('batch size')
plt.legend()
plt.show()

