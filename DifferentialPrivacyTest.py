import numpy as np
#algorithm_globals.random_seed = 42
import qiskit.quantum_info as qi
import copy
import argparse
import random
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

def calculate_rho_projection_ful_random_states(batch_size,dims):
    result_glob=[]
    result_glob_minus1=[]
    other_random_states=[]
    num_tests=10000
    dims=int(dims)
    num_total_states=int(batch_size*num_tests)
   
    for j in range(num_total_states):
        other_random_states.append(qi.DensityMatrix(qi.random_statevector(dims=dims)).data)
    other_random_states=np.array(other_random_states) 

    print('finished random states generation')
    for i in range(num_tests):
        rho_star=qi.DensityMatrix(qi.random_statevector(dims=dims)).data
        if i%100==0:
            print('finished', i)
        #numbers = random.sample(range(0, num_total_states), batch_size-1)
        current_states=other_random_states[i*batch_size:(i+1)*batch_size]
        #print(current_states)
        #print(rho_star)
        global_state=np.sum(current_states, axis=0)/(batch_size)+rho_star/batch_size
        #print(global_state)
        adj_state=np.sum(current_states, axis=0)/(batch_size-1)
        prob_glob=np.trace(np.matmul(global_state,rho_star))
        prob_glob_minus1=np.trace(np.matmul(adj_state,rho_star))
        #print(np.trace(np.matmul(current_states[0],rho_star)))
        #print(np.trace(np.matmul(rho_star,rho_star)))
        #print(np.trace(np.matmul(rho_star+current_states[0],rho_star)))
        #print(prob_glob,prob_glob_minus1)
        result_glob.append(prob_glob.real)
        result_glob_minus1.append(prob_glob_minus1.real)
    return result_glob,result_glob_minus1

def calculate_rho_projection_half_random_states(batch_size,dims):
    result_glob=[]
    result_glob_minus1=[]
    other_random_states=[]
    num_tests=100000
    dims=int(dims)
    num_total_states=int(batch_size*50)
   
    for j in range(num_total_states):
        other_random_states.append(qi.DensityMatrix(qi.random_statevector(dims=dims)).data)
    other_random_states=np.array(other_random_states) 

    print('finished random states generation')
    for i in range(num_tests):
        rho_star=qi.DensityMatrix(qi.random_statevector(dims=dims)).data
        if i%100==0:
            print('finished', i)
        numbers = random.sample(range(0, num_total_states), batch_size-1)
        current_states=list(other_random_states[numbers])
        #print(current_states)
        #print(rho_star)
        global_state=np.sum(current_states, axis=0)/(batch_size)+rho_star/batch_size
        #print(global_state)
        adj_state=np.sum(current_states, axis=0)/(batch_size-1)
        prob_glob=np.trace(np.matmul(global_state,rho_star))
        prob_glob_minus1=np.trace(np.matmul(adj_state,rho_star))
        #print(np.trace(np.matmul(current_states[0],rho_star)))
        #print(np.trace(np.matmul(rho_star,rho_star)))
        #print(np.trace(np.matmul(rho_star+current_states[0],rho_star)))
        #print(prob_glob,prob_glob_minus1)
        result_glob.append(prob_glob.real)
        result_glob_minus1.append(prob_glob_minus1.real)
    return result_glob,result_glob_minus1

def calculate_rho_projection_random_states(batch_size,dims):
    result_glob=[]
    result_glob_minus1=[]
    other_random_states=[]
    num_tests=100000
    dims=int(dims)
    num_total_states=int(batch_size*50)
   
    for j in range(num_total_states):
        other_random_states.append(qi.DensityMatrix(qi.random_statevector(dims=dims)).data)
    other_random_states=np.array(other_random_states)    
    rho_star=qi.DensityMatrix(qi.random_statevector(dims=dims)).data
    print('finished random states generation')
    for i in range(num_tests):
        if i%100==0:
            print('finished', i)
        numbers = random.sample(range(0, num_total_states), batch_size-1)
        current_states=list(other_random_states[numbers])
        #print(current_states)
        #print(rho_star)
        global_state=np.sum(current_states, axis=0)/(batch_size)+rho_star/batch_size
        #print(global_state)
        adj_state=np.sum(current_states, axis=0)/(batch_size-1)
        prob_glob=np.trace(np.matmul(global_state,rho_star))
        prob_glob_minus1=np.trace(np.matmul(adj_state,rho_star))
        #print(np.trace(np.matmul(current_states[0],rho_star)))
        #print(np.trace(np.matmul(rho_star,rho_star)))
        #print(np.trace(np.matmul(rho_star+current_states[0],rho_star)))
        #print(prob_glob,prob_glob_minus1)
        result_glob.append(prob_glob.real)
        result_glob_minus1.append(prob_glob_minus1.real)
    return result_glob,result_glob_minus1


###*** Below important ones ***###
def distribution_randomBatches(X,Y,number_of_tests,batch_size):

    #print('calculating output distribution using random batching')
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
    
    
    result_glob_minus1=[]
    result_glob=[]
    #for _ in range(1):
    #    random_class=random.randint(0, 1)
    print('number of tests:',len(X_perClass_DM[0]),len(X_perClass_DM[1]),int((len(X_perClass_DM[0])-1)/(batch_size-1)),int((len(X_perClass_DM[1])-1)/(batch_size-1)))
                
    for random_class in [0,1]:   
        class_x=X_perClass_DM[random_class]
        class_y=all_labels[random_class]
        #print('number of tests:',int((len(class_x)-1)/(batch_size-1)))
        #random_index=random.randint(0, len(class_x))
        for random_index in range(0,len(class_x)):
            rho_star=class_x[random_index]
            class_x_new=np.delete(np.array(class_x), random_index,axis=0)
            number_of_tests=int(len(class_x)/batch_size)
            #print(number_of_tests)
            #if number_of_tests<1:
            #    print('not enought data points for the batch size',number_of_tests)
                
            for batch_index in range(number_of_tests):


                #numbers = random.sample(range(0, len(class_x_new)), batch_size-1)
                if (batch_index+1)*(batch_size-1)<len(class_x_new):
                    numbers=np.arange(batch_index*(batch_size-1),(batch_index+1)*(batch_size-1),1)
                    current_states=list(class_x_new[numbers])
                    #print(len(current_states))
                    global_state=np.sum(current_states, axis=0)/(batch_size)+rho_star/batch_size
                    
                    assert(len(current_states)==batch_size-1)
                    adj_state=np.sum(current_states, axis=0)/(batch_size-1)
                    assert(qi.DensityMatrix(global_state).is_valid())
                    assert(qi.DensityMatrix(adj_state).is_valid())
                    prob_glob=np.trace(np.matmul(global_state,rho_star))
                    prob_glob_minus1=np.trace(np.matmul(adj_state,rho_star))
                    #print(prob_glob_minus1)

                    result_glob.append(prob_glob.real)
                    result_glob_minus1.append(prob_glob_minus1.real)
                else:
                    print('loosing one data point')

    print(len(result_glob))
    return result_glob,result_glob_minus1



def distribution_smartBatches(X,Y,number_of_batches,fidelity_matrix):
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
    result_glob_minus1=[]
    result_glob=[]
    for class_x,class_y,fidelity_matrix in zip(X_perClass_DM,all_labels,fidelity_matrices):
        batches_list=[[] for _ in range(number_of_batches)]
        clustering=SpectralClustering(n_clusters=number_of_batches,affinity='precomputed').fit(fidelity_matrix).labels_

        global_states=np.zeros((number_of_batches,dim,dim),dtype=np.complex128)
        number_per_cluster=np.zeros(number_of_batches)

        for x_current,cluster in zip(class_x,clustering):
            global_states[cluster]+=x_current
            batches_list[cluster].append(x_current)
            number_per_cluster[cluster]+=1
        print(np.mean(number_per_cluster))
        for cluster_index in range(number_of_batches):
            global_state_minus1=[]
            X_current=batches_list[cluster_index]
            global_state_all=global_states[cluster_index]
            for x_current in X_current:      
                rho_star=x_current          
                adj_state=copy.deepcopy(global_state_all)
                adj_state=adj_state-x_current
                adj_state=adj_state/(len(X_current)-1)
                global_state=global_state_all/len(X_current)

                
                
                assert(qi.DensityMatrix(global_state).is_valid())
                assert(qi.DensityMatrix(adj_state).is_valid())
                prob_glob=np.trace(np.matmul(global_state,rho_star))
                prob_glob_minus1=np.trace(np.matmul(adj_state,rho_star))
                #print(prob_glob_minus1)

                result_glob.append(prob_glob.real)
                result_glob_minus1.append(prob_glob_minus1.real)

    print(len(result_glob))
    return result_glob,result_glob_minus1







if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('-i','--input_data', required=True,help='the input data')

    parser.add_argument('--num_tests', required=False,help='number of tests, used for complete random case : argument random',default=1)

    parser.add_argument('--batch_size', type=int,required=False,help='size of a batch, used for random states test and randomBatches',default=10)
    parser.add_argument('--dims', required=False,help='dimension of the Density Matrices, used for random states test',default=64)
    
    parser.add_argument('--num_batches', required=False,help='number of batches for global quantum state, for multi-class dataset, it is the number of batches per class, used for smart Batch',default=1)
    parser.add_argument('--smart_batch', help='whether to use clustering based batches',action='store_true')
    parser.add_argument('--fidelity_matrix', required=False,help='precomputed fidelity matrix, used for smart Batching to speed up',default='../DP/fidelityMatrix_synSNPs256')
    
    parser.add_argument('--output', required=True,help='the output prefix')
    args = parser.parse_args()

    if args.input_data=='random':
        prob,prob_minus1=calculate_rho_projection_random_states(args.batch_size,args.dims)
        prob_half,prob_half_minus1=calculate_rho_projection_half_random_states(args.batch_size,args.dims)
        

    else:
        dataset=args.input_data

        X_input = np.loadtxt(dataset+'X.dat',dtype=np.complex128)
        y_input = np.loadtxt(dataset+'y.dat',dtype=np.complex128)
        y_input=np.real_if_close(y_input)

        split=int(0.7*len(X_input))
        Xtrain=X_input[:split]
        Xtest=X_input[split:]
        ytrain=y_input[:split]
        ytest=y_input[split:]

        smart_batch=args.smart_batch
        number_of_tests=int(args.num_tests)
        number_of_batches=int(args.num_batches)

        data_dimension=np.sqrt(len(Xtrain[0]))
        assert(data_dimension==int(data_dimension))
        data_dimension=int(data_dimension)

        #privacy_loss=similarity_neighbor_randomBatches(Xtrain,ytrain,number_of_batches)
        #privacy_loss=similarity_neighbor_smartBatches(Xtrain,ytrain,number_of_batches,args.fidelity_matrix)

        batch_size=args.batch_size
        for dd in range(1):
            print(len(Xtrain))
            #prob,prob_minus1=distribution_randomBatches(Xtrain,ytrain,number_of_tests,batch_size)
            prob,prob_minus1=distribution_smartBatches(Xtrain,ytrain,number_of_batches,args.fidelity_matrix)
            #print(prob_minus1)
        
            from scipy.stats import norm
            mu, std = norm.fit(prob)
            mu_adj, std_adj = norm.fit(prob_minus1)
            print(mu,mu_adj,std,std_adj)#,abs(mu-mu_adj)/std)
    #exit()
    
    ranges=(0.2,0.8)
    resolution=0.001
    range_size=ranges[1]-ranges[0]
    num_bin=int(range_size/resolution)
    #histo1_all=np.histogram(prob, bins=num_bin, range=ranges,density=False)[0]
    #histo1_adj=np.histogram(prob_minus1, bins=num_bin, range=ranges,density=False)[0]
    #total_events=sum(histo1_all)
    #threshold=total_events*0.001
    #print(threshold)
    #epsilon_list=[]
    #for i,j in zip(histo1_all,histo1_adj):
    #    if i>threshold and j>threshold:
            #print(i/j,i,j)
    #        epsilon_list.append(i/j)
    #print(max(epsilon_list))

    total_result=np.array([prob,prob_minus1])
    np.savetxt(args.output+'withinDataset_smart_LoopOverEachState_rhostar_projection_prob.dat',total_result)
    bins=np.arange(min(prob_minus1)-0.1,max(prob)+0.1,resolution)


    plt.hist(prob,bins=bins,alpha=0.5,density=True,label='global',color='blue')
    plt.hist(prob_minus1,bins=bins,alpha=0.5,density=True,label='adj',color='orange')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf, 'r--', label='Fitted global PDF',color='blue')
    pdf_adj = norm.pdf(x, mu_adj, std_adj)
    plt.plot(x, pdf_adj, 'r--', label='Fitted adj PDF',color='orange')
    plt.legend()
    #plt.hist(prob_half,alpha=0.5,bins=bins,histtype='step',density=True)
    #plt.hist(prob_half_minus1,alpha=0.5,bins=bins,histtype='step',density=True)
    plt.show()
        










