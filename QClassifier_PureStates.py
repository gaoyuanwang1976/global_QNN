import numpy as np
from qiskit.circuit import ParameterVector,QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.circuit.library import RealAmplitudes,ZZFeatureMap,RXGate
from qiskit_machine_learning.circuit.library import QNNCircuit
import preprocessing
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 42

import copy
import argparse
import global_core

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=5)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of alternating R_ZX and R_XX layers in the QNN', default=3)
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('-o','--optimizer', required=False, type=str, help='determines the Qiskit optimizer used in qnn', default='cobyla')
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('-n','--num_auxiliary_qubits', required=False,help='number of auxiliary qubits',default=0)
    parser.add_argument('--best_score', required=False,help='the best training score for existing model, to be used in combinatino with --exisiting model',default=None)
    parser.add_argument('-i','--input_data', required=True,help='the input data')
    
    args = parser.parse_args()
    parsed_shots=args.shots
    n_layers = args.num_layers
    n_extra_qubits=int(args.num_auxiliary_qubits)
    n_epochs=args.epochs
    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)

    ## for both cobyla and spsa, derivative is the objective is not used
    if args.optimizer.lower() == 'cobyla':
        optimizer = COBYLA(maxiter=50)
    elif args.optimizer.lower() == 'spsa':
        optimizer = SPSA(maxiter=100)
    else:
        print("problem with parsing optimizer, defaulting to COBYLA")
        optimizer = COBYLA(maxiter=100)

    dataset=args.input_data
    X_input = np.loadtxt(dataset+'X.dat',dtype=np.complex128)

    y_input_original = np.loadtxt(dataset+'y.dat',dtype=np.complex128)
    y_input=preprocessing.unify_y_label(y_input_original)

    X_input=np.real_if_close(X_input)
    #y_input=np.real_if_close(y_input)
    X_input,y_input=preprocessing.alternating_data(X_input,y_input)

    data_length=100#len(X_input)
    split=int(0.7*len(X_input))
    Xtrain=X_input[:split]
    Xtest=X_input[split:]
    ytrain=y_input[:split]
    ytest=y_input[split:]
    
    data_dimension=int(len(Xtrain[0]))
 
    
###############################
##### amplitude embedding #####
###############################
    n_features = data_dimension
    x_params = ParameterVector('x',n_features)
    n_qubit=np.sqrt(data_dimension)
    assert(n_qubit==int(n_qubit))
    n_qubit=int(n_qubit)
    feature_map = RawFeatureVector(feature_dimension=n_features) # amplitude encoding 
    #feature_map.assign_parameters(x_params)
    ansatz = RealAmplitudes(num_qubits=n_qubit)
    #theta_params = ParameterVector('theta', len(ansatz.parameters))
    #ansatz.assign_parameters(theta_params)
    qc = QuantumCircuit(n_qubit)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

############################
###### QNN Classifier ######
############################

#### callback_function stores weights and loss during fitting
    def callback_function(current_weight, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        weights.append(current_weight)

    ### ansatz has to be equal to the ansatz of the circuit, i.e., everything after the amplitude encoding.
    circuit_qnn = EstimatorQNN(circuit=qc,input_params=feature_map.parameters,weight_params=ansatz.parameters,input_gradients=False)
    weights=[1]*len(ansatz.parameters)
    circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,optimizer=optimizer,loss= 'absolute_error',warm_start=True,callback=callback_function)

##########################
####### Fitting ##########
##########################    
    print('training...')

    objective_func_vals = [] ### empty list to store returned values of the callback_function
    weights = [] ### empty list to store returned values of the callback_function

#### fit classifier to data ####
    best_train_score=0
    for epoch in range(n_epochs):
        circuit_classifier.fit(Xtrain,ytrain)
        #print(*circuit_classifier.predict(Xtrain))
        train_score=circuit_classifier.score(Xtrain,ytrain)
        test_score=circuit_classifier.score(Xtest,ytest)
        print(epoch,train_score,test_score)
        #objective_func_vals = []
        #circuit_classifier.fit(Xtrain_glob,ytrain_glob)




