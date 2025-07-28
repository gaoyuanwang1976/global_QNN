import numpy as np
#algorithm_globals.random_seed = 42
import qiskit.quantum_info as qi
import copy
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def compute_fidelity_matrix(X):
    num_data=len(X)
    fidelity_matrix=np.zeros((num_data,num_data))
    for index_i,i in enumerate(X):
        print('index i',index_i)
        fidelity_matrix[index_i,index_i]=1
        for index_j in range(index_i+1,num_data):
            j=X[index_j]
            fidelity=qi.state_fidelity(i,j)
            fidelity_matrix[index_i,index_j]=fidelity
            fidelity_matrix[index_j,index_i]=fidelity

    #np.savetxt('fidelityMatrix_PsychENCODE256.dat',fidelity_matrix)
    return fidelity_matrix

def load_fidelity_matrix(filename):
    class0=np.loadtxt(filename+'0.0.dat')
    class1=np.loadtxt(filename+'1.0.dat')
    print(class0.shape)
    return [class0,class1]


def similarity_neighbor_randomBatches(X,Y,number_of_batches):
    print('using random batching')
    total_number_of_data=len(X)
    all_labels=sorted(set(Y))
    number_of_classes=len(all_labels)
    dim=np.sqrt(len(X[0]))
    assert(dim==int(dim))
    dim=int(dim)
    batch_size_list_all_label=[]

    X_DM=[]
    X_perClass_DM=[[] for _ in range(number_of_classes)]
    X_glob=[]
    y_glob=[]
    for data_i,y in zip(X,Y):
        x_dm=np.reshape(data_i,shape=(dim,dim))
        X_DM.append(x_dm)
        current_label_index=all_labels.index(y)
        X_perClass_DM[current_label_index].append(x_dm)
    
    privacy_loss=[]

    for class_x,class_y in zip(X_perClass_DM,all_labels):
        batch_size=int(len(class_x)/number_of_batches)
        batch_size_list=[]
        remaining=len(class_x)-batch_size*number_of_batches
        assert(remaining>=0)
        batch_offset=[]
        offset=0

        index_remaining=0
        for index_batch in range(number_of_batches):
            if index_remaining < remaining:
                batch_size_list.append(batch_size+1)
                index_remaining+=1

            else:
                batch_size_list.append(batch_size)
            batch_offset.append(offset)
            offset+=batch_size_list[-1]
        print(batch_offset,batch_size_list)
        assert(sum(batch_size_list)==len(class_x))

        for batch_index in range(number_of_batches):
            X_current=class_x[batch_offset[batch_index]:batch_offset[batch_index]+batch_size_list[batch_index]]
            global_state_all=np.zeros((dim,dim),dtype=np.complex128)

            for x_current in X_current:
                global_state_all+=x_current
            global_state_minus1=[]
            #if len(X_current)==1:
            #print('X_current length',len(X_current))
            for x_current in X_current:
                global_state_tmp=copy.deepcopy(global_state_all)
                global_state_tmp=global_state_tmp-x_current
                global_state_tmp=global_state_tmp/(len(X_current)-1)
                global_state_minus1.append(qi.DensityMatrix(global_state_tmp))
            global_state_all=qi.DensityMatrix(global_state_all/len(X_current))
            for neighbor in global_state_minus1:
                #print(global_state.is_valid(),neighbor.is_valid())
                #print(neighbor.trace())
                fidelity_tmp=qi.state_fidelity(global_state_all,neighbor)
                privacy_loss.append(1./fidelity_tmp)

    return privacy_loss

def distribution_randomBatches(X,Y,number_of_batches):

    print('calculating output distribution using random batching')
    total_number_of_data=len(X)
    all_labels=sorted(set(Y))
    number_of_classes=len(all_labels)
    dim=np.sqrt(len(X[0]))
    assert(dim==int(dim))
    dim=int(dim)
    batch_size_list_all_label=[]

    X_DM=[]
    X_perClass_DM=[[] for _ in range(number_of_classes)]
    X_glob=[]
    y_glob=[]
    for data_i,y in zip(X,Y):
        x_dm=np.reshape(data_i,shape=(dim,dim))
        X_DM.append(x_dm)
        current_label_index=all_labels.index(y)
        X_perClass_DM[current_label_index].append(x_dm)
    
    #privacy_loss=[]
    result_glob_minus1=[]
    result_glob=[]
    for class_x,class_y in zip(X_perClass_DM,all_labels):
        batch_size=int(len(class_x)/number_of_batches)
        batch_size_list=[]
        remaining=len(class_x)-batch_size*number_of_batches
        assert(remaining>=0)
        batch_offset=[]
        offset=0

        index_remaining=0
        for index_batch in range(number_of_batches):
            if index_remaining < remaining:
                batch_size_list.append(batch_size+1)
                index_remaining+=1

            else:
                batch_size_list.append(batch_size)
            batch_offset.append(offset)
            offset+=batch_size_list[-1]
        #print(batch_offset,batch_size_list)
        assert(sum(batch_size_list)==len(class_x))

        for batch_index in range(number_of_batches):
            X_current=class_x[batch_offset[batch_index]:batch_offset[batch_index]+batch_size_list[batch_index]]
            global_state_all=np.zeros((dim,dim),dtype=np.complex128)

            for x_current in X_current:
                global_state_all+=x_current
            global_state_minus1=[]

            for x_current in X_current:
                global_state_tmp=copy.deepcopy(global_state_all)
                global_state_tmp=global_state_tmp-x_current
                global_state_tmp=global_state_tmp/(len(X_current)-1)
                global_state_minus1.append(qi.DensityMatrix(global_state_tmp))
            global_state_all=qi.DensityMatrix(global_state_all/len(X_current))
            for neighbor,x_current in zip(global_state_minus1,X_current):
                x_current_DM=qi.DensityMatrix(x_current)
                #prob_glob=global_state_all.expand(x_current_DM).trace()
                #prob_glob_minus1=neighbor.expand(x_current_DM).trace()
                prob_glob=np.trace(np.matmul(global_state_all.data,x_current))
                prob_glob_minus1=np.trace(np.matmul(neighbor.data,x_current))
                result_glob.append(prob_glob.real)
                result_glob_minus1.append(prob_glob_minus1.real)
                #print(prob_glob,prob_glob_minus1)
    return result_glob,result_glob_minus1

def similarity_neighbor_smartBatches(X,Y,number_of_batches,fidelity_matrix):
    print('using smart batching')
    total_number_of_data=len(X)
    all_labels=sorted(set(Y))
    number_of_classes=len(all_labels)
    dim=np.sqrt(len(X[0]))
    assert(dim==int(dim))
    dim=int(dim)

    batch_size_list_all_label=[]

    X_DM=[]
    X_perClass_DM=[[] for _ in range(number_of_classes)]
    X_glob=[]
    y_glob=[]
    for data_i,y in zip(X,Y):

        x_dm=np.reshape(data_i,shape=(dim,dim))
        X_DM.append(x_dm)
        current_label_index=all_labels.index(y)
        X_perClass_DM[current_label_index].append(x_dm)

    privacy_loss=[]
    fidelity_matrices=load_fidelity_matrix(fidelity_matrix)

    for class_x,class_y,fidelity_matrix in zip(X_perClass_DM,all_labels,fidelity_matrices):
        batches_list=[[] for _ in range(number_of_batches)]
        clustering=SpectralClustering(n_clusters=number_of_batches,affinity='precomputed').fit(fidelity_matrix).labels_

        global_states=np.zeros((number_of_batches,dim,dim),dtype=np.complex128)
        number_per_cluster=np.zeros(number_of_batches)

        for x_current,cluster in zip(class_x,clustering):
            global_states[cluster]+=x_current
            batches_list[cluster].append(x_current)
            number_per_cluster[cluster]+=1

        for cluster_index in range(number_of_batches):
            global_state_minus1=[]
            X_current=batches_list[cluster_index]
            global_state_all=global_states[cluster_index]
            for x_current in X_current:
                
                global_state_tmp=copy.deepcopy(global_state_all)
                global_state_tmp=global_state_tmp-x_current
                global_state_tmp=global_state_tmp/(len(X_current)-1)
                global_state_minus1.append(qi.DensityMatrix(global_state_tmp))
            global_state_all=qi.DensityMatrix(global_state_all/len(X_current))
            assert(number_per_cluster[cluster_index]==len(X_current))
            for neighbor in global_state_minus1:
                fidelity_tmp=qi.state_fidelity(global_state_all,neighbor)
                privacy_loss.append(1./fidelity_tmp)

    return privacy_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('-i','--input_data', required=True,help='the input data')
    parser.add_argument('--num_batches', required=False,help='number of batches for global quantum state, for multi-class dataset, it is the nnumber of batches per class',default=1)
    parser.add_argument('--smart_batch', help='whether to use clustering based batches',action='store_true')
    parser.add_argument('--fidelity_matrix', required=False,help='whether to use clustering based batches',default='../DP/fidelityMatrix_synSNPs256')
    parser.add_argument('--output', required=True,help='the output prefix')
    args = parser.parse_args()
    dataset=args.input_data

    X_input = np.loadtxt(dataset+'X.dat',dtype=np.complex128)
    y_input = np.loadtxt(dataset+'y.dat',dtype=np.complex128)
    y_input=np.real_if_close(y_input)
    print(len(X_input[0]),args.fidelity_matrix)
   
    split=int(0.7*len(X_input))
    Xtrain=X_input[:split]
    Xtest=X_input[split:]
    ytrain=y_input[:split]
    ytest=y_input[split:]

    print(len(Xtrain),len(Xtest),len(X_input))
    smart_batch=args.smart_batch
    number_of_batches=int(args.num_batches)


    data_dimension=np.sqrt(len(Xtrain[0]))
    assert(data_dimension==int(data_dimension))
    data_dimension=int(data_dimension)

    #privacy_loss=similarity_neighbor_randomBatches(Xtrain,ytrain,number_of_batches)
    #privacy_loss=similarity_neighbor_smartBatches(Xtrain,ytrain,number_of_batches,args.fidelity_matrix)
    prob,prob_minus1=distribution_randomBatches(Xtrain,ytrain,number_of_batches)
    total_result=np.array([prob,prob_minus1])
    print(total_result.shape)
    np.savetxt(args.output+'_rhostar_projection_prob.dat',total_result)
    plt.hist(prob)
    plt.hist(prob_minus1)
    plt.show()
    
    #print('max',max(privacy_loss))
    #print('mean',np.mean(privacy_loss))









