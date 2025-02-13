import numpy as np
from qiskit.quantum_info import DensityMatrix

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return np.array(v/norm)

def normalize_angle(import_name,genomics_dataset,X):
    maxData = get_max_data(import_name,genomics_dataset)
    minData = get_min_data(import_name,genomics_dataset)
    X=(X-minData)/(maxData-minData)*np.pi
    return X

def normalize_amplitude(X):
    X_norm=[]
    for x in X:
        X_norm.append(normalize(x))
    return np.array(X_norm)

def construct_required_states_DM(X,Y):

    dim=np.sqrt(len(X[0]))
    assert(dim==int(dim))
    dim=int(dim)
    X_DM=[]
    X_one_DM=[]
    X_zero_DM=[]
    X_one_glob=np.zeros((dim,dim))
    X_zero_glob=np.zeros((dim,dim))
    for data_i,y in zip(X,Y):
        x_dm=np.reshape(data_i,shape=(dim,dim))
        #x_dm=np.real_if_close(x_dm)
        X_DM.append(x_dm)

        if y==1:
            X_one_DM.append(x_dm)
            X_one_glob=X_one_glob+x_dm
        elif y==0 or y==-1:
            X_zero_DM.append(x_dm)
            X_zero_glob=X_zero_glob+x_dm
        else:
            raise Error("Unknown label")
    X_one_glob=X_one_glob/len(X_one_DM)
    X_zero_glob=X_zero_glob/len(X_zero_DM)   
    X_glob=[X_one_glob,X_zero_glob]
    y_glob=[1,0]
    X_one_glob=[X_one_glob]
    X_zero_glob=[X_zero_glob]
    return np.array(X_DM),np.array(X_one_DM),np.array(X_zero_DM),np.array(X_one_glob),np.array(X_zero_glob),np.array(X_glob),np.array(y_glob)

def construct_required_states(X,Y):
    dim=len(X[0])
    X_DM=[]
    X_one_DM=[]
    X_zero_DM=[]
    X_one_glob=np.zeros((dim,dim))
    X_zero_glob=np.zeros((dim,dim))
    for data_i,y in zip(X,Y):
        x_dm=DensityMatrix(data_i).data
        x_dm=np.complex128(np.real_if_close(x_dm))
        X_DM.append(x_dm)

        if y==1:
            X_one_DM.append(x_dm)
            X_one_glob=X_one_glob+x_dm
        elif y==0 or y==-1:
            X_zero_DM.append(x_dm)
            X_zero_glob=X_zero_glob+x_dm
        else:
            raise ValueError("Unknown label")
    X_one_glob=X_one_glob/len(X_one_DM)
    X_zero_glob=X_zero_glob/len(X_zero_DM)   
    X_glob=[X_one_glob,X_zero_glob]
    y_glob=[1,0]
    X_one_glob=[X_one_glob]
    X_zero_glob=[X_zero_glob]


    return np.array(X_DM),np.array(X_one_DM),np.array(X_zero_DM),np.array(X_one_glob),np.array(X_zero_glob),np.array(X_glob),np.array(y_glob)


def construct_required_states_EQ(X,Y):
    dim=len(X[0])
    X_DM=[] ### although called DM, these are actually state vectors because EstimatQNN only takes state vectors as inputs
    X_one_DM=[]
    X_zero_DM=[]
    for data_i,y in zip(X,Y):
        if y==1:
            X_one_DM.append(data_i)
        elif y==0 or y==-1:
            X_zero_DM.append(data_i)

        else:
            raise ValueError("Unknown label")

    return X,np.array(X_one_DM),np.array(X_zero_DM)

def alternating_data(X,y):
    one_data=[]
    zero_data=[]
    
    for data,label in zip(X,y):
        if label==1:
            one_data.append(data)
        elif label==0 or label==-1:
            negative_label=label
            zero_data.append(data)
        else:
            raise ValueError('Unknown label.')
    alter_data=[]
    alter_label=[]
    for zero,one in zip(zero_data,one_data):
        alter_data.append(zero)
        alter_label.append(negative_label)
        alter_data.append(one)
        alter_label.append(1)
    return np.array(alter_data),np.array(alter_label)

def unify_y_label(y_input_original):
    y_input=[]
    for y in y_input_original:
        if y==1:
            y_input.append(y)
        elif y==-1:
            y_input.append(y)
        elif y==0:
            y_input.append(-1)
        else:
            raise ValueError("unknown label")
    y_input=np.array(y_input)
    y_input=np.real_if_close(y_input)
    return y_input

        

