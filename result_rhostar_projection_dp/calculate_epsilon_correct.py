import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
input_list=['256']
delta=0.05
#data=np.loadtxt('smartBatching/smartBatching_mu_muadj_std_stdadj_epsilon_'+input_list[0]+'SNPs.dat')
data=np.loadtxt(sys.argv[1])

def first_method(input_list,delta):
    result_list=[]

    for take in range(len(data)):
        bs=data.T[0][take]
        mu=data.T[1][take]
        mu_adj=data.T[2][take]
        sigma=data.T[3][take]
        sigma_adj=data.T[4][take]

        l=[]
        for _ in range(10000):
            x = np.random.normal(mu, sigma)
            g1_x=norm.pdf(x, mu, sigma)
            g2_x=norm.pdf(x, mu_adj, sigma_adj)
            l.append(np.log(g1_x/g2_x))
        #print(bs,np.percentile(l, 99.5))
        epsilon=np.percentile(l, 95)
        #epsilon = norm.ppf(0.95, loc=mu, scale=sigma)
        larger_count=0
        for i in l:
            if i >= epsilon:
                larger_count+=1
        #print(larger_count/len(l))
        result_list.append(epsilon)
        print('average case',epsilon)
    #print(result_list)
    return(result_list)


def second_method(input_list,result_list):
    
        

    for take in range(len(data)):
        bs=data.T[0][take]
        mu=data.T[1][take]
        mu_adj=data.T[2][take]
        sigma=data.T[3][take]
        sigma_adj=data.T[4][take]
        #print(mu, mu_adj,sigma, sigma_adj)
        #x = np.linspace(-10, 10, 10000)
        x = np.random.normal(mu, sigma,10000)
        pdf = norm.pdf(x, mu, sigma)
        
        pdf_adj=norm.pdf(x, mu_adj, sigma_adj)
        epsilon=result_list[take]
        privacy_losses = np.log(pdf / pdf_adj)
        delta_estimate = np.mean(privacy_losses > epsilon)
        delta=max(pdf-np.exp(epsilon)*pdf_adj)
        print('delta worst vs mean',delta,delta_estimate)

def compute_worst_case_epsilon(input_list, delta):
    
        
    epsilon_list=[]
    for take in range(len(data)):
        bs=data.T[0][take]
        mu=data.T[1][take]
        mu_adj=data.T[2][take]
        sigma=data.T[3][take]
        sigma_adj=data.T[4][take]
        #print(mu, mu_adj,sigma, sigma_adj)
        x = np.linspace(-1, 2, 10000)
        #x = np.random.normal(mu, sigma,50000)
        pdf = norm.pdf(x, mu, sigma)
        
        pdf_adj=norm.pdf(x, mu_adj, sigma_adj)
        pdf_P = np.array(pdf)
        pdf_Q = np.array(pdf_adj)

        # Avoid division by zero or log of negative numbers
        valid = (pdf_Q > 0) & (pdf_P > delta)
        #valid=range(len(x))
        plt.plot(x[valid],pdf_P[valid]-delta,alpha=0.2,marker='s')
        plt.plot(x[valid],pdf_Q[valid],linestyle='dashed')
        #plt.plot(pdf_P/pdf_Q)
        ratio = (pdf_P[valid] - delta) / pdf_Q[valid]
        plt.plot(x[valid],ratio)
        # Compute the supremum
        epsilon = np.max(np.log(ratio))
        print('worst case',bs,epsilon)
        epsilon_list.append(epsilon)
    #plt.show()
    return epsilon_list

def compute_worst_case_epsilon_switched(input_list, delta):
    
        
    epsilon_list=[]
    for take in range(1):#(len(data)):
        bs=data.T[0][take]
        mu=data.T[1][take]
        mu_adj=data.T[2][take]
        sigma=data.T[3][take]
        sigma_adj=data.T[4][take]
        #print(mu, mu_adj,sigma, sigma_adj)
        x = np.linspace(-1, 2, 10000)
        #x = np.random.normal(mu_adj, sigma_adj,50000)
        pdf = norm.pdf(x, mu, sigma)
        
        pdf_adj=norm.pdf(x, mu_adj, sigma_adj)
        pdf_P = np.array(pdf_adj)
        pdf_Q = np.array(pdf)

        # Avoid division by zero or log of negative numbers
        valid = (pdf_Q > 0) & (pdf_P > delta)
        
        plt.plot(x[valid],pdf_P[valid]-delta,alpha=0.2,marker='s')
        plt.plot(x[valid],pdf_Q[valid],linestyle='dashed',marker='o')
        plt.plot(x,pdf_P)
        plt.plot(x,pdf_Q)
        #plt.plot(pdf_P/pdf_Q)
        ratio = (pdf_P[valid] - delta) / pdf_Q[valid]
        #print(ratio)
        # Compute the supremum
        epsilon = np.max(np.log(ratio))
        print('worst case',bs,epsilon)
        epsilon_list.append(epsilon)
    plt.show()
    return epsilon_list


#result_list=first_method(input_list,delta)
#second_method(input_list,result_list)
epsilon=compute_worst_case_epsilon(input_list,delta)

#second_method(input_list,epsilon)