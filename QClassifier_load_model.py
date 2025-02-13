
import numpy as np
from qiskit.circuit import ParameterVector,QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.quantum_info import SparsePauliOp,Operator
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.circuit.library import RealAmplitudes,ZZFeatureMap,RXGate,EfficientSU2
from qiskit_machine_learning.circuit.library import QNNCircuit
import preprocessing
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = 42
import test_core
import copy
import argparse
import global_core
import ray
#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('-i','--input_data', required=True,help='the input data')
    parser.add_argument('--input_type', required=False,help='input type, density matrix or vector',default='density_matrix')
    parser.add_argument('-m','--input_model', required=True,help='the pretrained model name')
    args = parser.parse_args()

    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)


    dataset=args.input_data
    model_name=args.input_model
    X_input = np.loadtxt(dataset+'X.dat',dtype=np.complex128)
    y_input_original = np.loadtxt(dataset+'y.dat',dtype=np.complex128)
    y_input=preprocessing.unify_y_label(y_input_original) #change output (0,1) to (-1,1)
    y_input=np.real_if_close(y_input)
    X_input,y_input=preprocessing.alternating_data(X_input,y_input)

    split=int(0.7*len(X_input))
    Xtrain=X_input[:split]
    Xtest=X_input[split:]
    ytrain=y_input[:split]
    ytest=y_input[split:]

    if args.input_type=='density_matrix':
        print('using density matrix')
        data_dimension=np.sqrt(len(Xtrain[0]))
        assert(data_dimension==int(data_dimension))
        data_dimension=int(data_dimension)
        Xtrain_DM,Xtrain_one_DM,Xtrain_zero_DM,Xtrain_one_glob,Xtrain_zero_glob,Xtrain_glob,ytrain_glob=preprocessing.construct_required_states_DM(Xtrain,ytrain)
        Xtest_DM,Xtest_one_DM,Xtest_zero_DM,Xtest_one_glob,Xtest_zero_glob,Xtest_glob,ytest_glob=preprocessing.construct_required_states_DM(Xtest,ytest)

    best_model=global_core.MixedState_NNClassifier.load(model_name)
    best_test=best_model.score(Xtest_DM,ytest)
    predict_one_prob=(best_model.predict_prob(Xtest_DM)+1)*1./2
    auc=roc_auc_score(ytest,predict_one_prob)
    print(np.round(auc,3),np.round(best_test,3))