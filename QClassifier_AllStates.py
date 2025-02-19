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
import matplotlib.pyplot as plt

# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=10)
    parser.add_argument('-p','--patience', required=False, type=int, help='number of patience for early stop', default=5)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of layers in the QNN', default=3)
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('-o','--optimizer', required=False, type=str, help='determines the Qiskit optimizer used in qnn', default='cobyla')
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('-n','--num_auxiliary_qubits', required=False,help='number of auxiliary qubits',default=0)
    parser.add_argument('--best_score', required=False,help='the best training score for existing model, to be used in combinatino with --exisiting model',default=None)
    parser.add_argument('-i','--input_data', required=True,help='the input data')
    parser.add_argument('--input_type', required=False,help='input type, density matrix or vector',default='density_matrix')
    parser.add_argument('--global_state', help='whether to use global density matrix for training',action='store_true')
    parser.add_argument('--skip_final_rotation', required=False,help='whether to skip the final rotation layer in the RealAmplitude ansatz',default=False)
    parser.add_argument('--output', required=True,help='output name to save the best model')
    args = parser.parse_args()


    parsed_shots=args.shots
    n_layers = args.num_layers
    n_extra_qubits=int(args.num_auxiliary_qubits)
    n_epochs=args.epochs
    n_patience=args.patience
    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)
    skip_final_rotation=args.skip_final_rotation

    ## for both cobyla and spsa, derivative is the objective is not used
    if args.optimizer.lower() == 'cobyla':
        optimizer = COBYLA(maxiter=200)
    elif args.optimizer.lower() == 'spsa':
        optimizer = SPSA(maxiter=200)
    else:
        print("problem with parsing optimizer, defaulting to COBYLA")
        optimizer = COBYLA(maxiter=200)

    dataset=args.input_data
    output_name=args.output

    X_input = np.loadtxt(dataset+'X.dat',dtype=np.complex128)
    y_input = np.loadtxt(dataset+'y.dat',dtype=np.complex128)
    y_input=np.real_if_close(y_input)

    '''
    X_input = np.loadtxt(dataset+'X.dat',dtype=np.complex128)
    y_input_original = np.loadtxt(dataset+'y.dat',dtype=np.complex128)
    y_input=preprocessing.unify_y_label(y_input_original) #unify label to {0,1}
    y_input=np.real_if_close(y_input)

    np.random.seed(50)
    shuffle_idx = np.random.permutation(len(X_input))
    #print(shuffle_idx[:3])
    X_input = X_input[shuffle_idx]
    y_input = y_input[shuffle_idx]

    X_input,y_input=preprocessing.alternating_data(X_input,y_input)
    '''
    #np.savetxt('synSNPs_dataset_shuffled/GenoType123_top_512SNPs_DM_shuffle_X.dat',X_input)
    #np.savetxt('synSNPs_dataset_shuffled/GenoType123_top_512SNPs_DM_shuffle_y.dat',y_input)
    
    #exit()
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

    elif args.input_type=='vector':
        data_dimension=int(len(Xtrain[0]))
        Xtrain=preprocessing.normalize_amplitude(Xtrain)
        Xtest=preprocessing.normalize_amplitude(Xtest)
        Xtrain_DM,Xtrain_one_DM,Xtrain_zero_DM,Xtrain_one_glob,Xtrain_zero_glob,Xtrain_glob,ytrain_glob=preprocessing.construct_required_states(Xtrain,ytrain)
        Xtest_DM,Xtest_one_DM,Xtest_zero_DM,Xtest_one_glob,Xtest_zero_glob,Xtest_glob,ytest_glob=preprocessing.construct_required_states(Xtest,ytest)

    elif args.input_type=='vector_estimatorqnn':
        print('using vector_estimatorqnn')
        data_dimension=int(len(Xtrain[0]))
        Xtrain=preprocessing.normalize_amplitude(Xtrain)
        Xtest=preprocessing.normalize_amplitude(Xtest)
        Xtrain_DM,Xtrain_one_DM,Xtrain_zero_DM=preprocessing.construct_required_states_EQ(Xtrain,ytrain) ## no global states can be constructed 
        Xtest_DM,Xtest_one_DM,Xtest_zero_DM=preprocessing.construct_required_states_EQ(Xtest,ytest)
        Xtrain_DM=np.real_if_close(Xtrain_DM)
        Xtest_DM=np.real_if_close(Xtest_DM)

###############################
##### amplitude embedding #####
###############################
    n_features = data_dimension
    x_params = ParameterVector('x',n_features)
    import math
    n_qubit=math.log2(data_dimension)#np.sqrt(data_dimension)
    assert(n_qubit==int(n_qubit))
    n_qubit=int(n_qubit)
    feature_map = RawFeatureVector(feature_dimension=n_features) # amplitude encoding 
    ansatz = RealAmplitudes(num_qubits=n_qubit,reps=n_layers,insert_barriers=True,entanglement='sca',flatten=True,skip_final_rotation_layer=skip_final_rotation)
    print('number of parameters in the ansatz',len(ansatz.parameters))
    #ansatz = EfficientSU2(4, su2_gates=['rz'], entanglement='circular', reps=n_layers, flatten=False)

    #theta_params = ParameterVector('theta', len(ansatz.parameters))
    #ansatz.assign_parameters(theta_params)
    qc = QuantumCircuit(n_qubit)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    #print(ansatz.decompose())
############################
###### QNN Classifier ######
############################

#### callback_function stores weights and loss during fitting
    def callback_function(current_weight, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        weights.append(current_weight)
    #weights_ini=np.random.normal(size=len(ansatz.parameters))
    weights_ini=[0]*len(ansatz.parameters)
    ### ansatz has to be equal to the ansatz of the circuit, i.e., everything after the amplitude encoding.
    #circuit_qnn = global_core.MixedState_EstimatorQNN(circuit=qc,ansatz=ansatz,input_params=feature_map.parameters,weight_params=ansatz.parameters,input_gradients=False)
    if args.input_type=='vector_estimatorqnn':
        circuit_qnn = EstimatorQNN(circuit=qc,input_params=feature_map.parameters,weight_params=ansatz.parameters)
        circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,optimizer=optimizer,loss= 'absolute_error',warm_start=True,callback=callback_function)#,initial_point=weights_ini)
    elif args.input_type=='density_matrix':
        circuit_qnn=global_core.MixedState_EstimatorQNN(circuit=qc,ansatz=ansatz,input_params=feature_map.parameters,weight_params=ansatz.parameters)
        circuit_classifier = global_core.MixedState_NNClassifier(neural_network=circuit_qnn,optimizer=optimizer,loss= 'absolute_error',warm_start=True,callback=callback_function,initial_point=weights_ini)
    else:
        raise ValueError("Unknown input type")

##########################
####### Fitting ##########
##########################    
    print('training...')

    objective_func_vals = [] ### empty list to store returned values of the callback_function
    weights = [] ### empty list to store returned values of the callback_function


#### fit classifier to data ####
    best_train_score=0
    best_epoch=0
    best_test=0
    best_model=None
    patience=0
    last_train_score=0
    for epoch in range(n_epochs):
        if args.global_state is True:
            if epoch==0:
                print('Training on global objective function')
            circuit_classifier.fit(Xtrain_glob,ytrain_glob)
        else:
            circuit_classifier.fit(Xtrain_DM,ytrain)

        train_score=circuit_classifier.score(Xtrain_DM,ytrain)
        test_score=circuit_classifier.score(Xtest_DM,ytest)

        print(epoch,train_score,test_score)
        
        if train_score>best_train_score:
            best_train_score=train_score
            best_epoch=epoch
            best_test=test_score
            best_model=copy.deepcopy(circuit_classifier)
    
        if train_score<=last_train_score:
            patience+=1
        else:
            patience=0
        last_train_score=train_score
        if patience > n_patience:
            print('stop training due to patience')
            break

    predict_one_prob=best_model.predict_prob(Xtest_DM)
    predict_one_prob_train=best_model.predict_prob(Xtrain_DM)
    print('numbers test',np.sum(predict_one_prob >0.5),np.sum(predict_one_prob <0.5),np.mean(predict_one_prob))
    print('numbers train',np.sum(predict_one_prob_train >0.5),np.sum(predict_one_prob_train <0.5),np.mean(predict_one_prob_train))

    
    auc=roc_auc_score(ytest,predict_one_prob)
    #fpr, tpr, _ = roc_curve(ytest, predict_one_prob)
    #plt.figure(figsize=(6, 6))
    #plt.plot(fpr, tpr, color='blue', lw=2)
    
    print(best_epoch,np.round(auc,3),np.round(best_test,3))
    if output_name!='None':
        best_model.save(output_name+".model")
    #plt.show()






