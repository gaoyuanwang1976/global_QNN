import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')

#plt.axis([0, 350, 0, 10])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Privacy Parameter ε', fontsize=12, fontweight='bold')
plt.xlabel('Average Batch Size', fontsize=12, fontweight='bold')
plt.title('Privacy - 128 and 256 Synthetic SNPs', 
          fontsize=14, fontweight='bold', pad=20)

input_list=['128']
color_list=['salmon','#e74c3c']
color_list_2=['cyan','#3498db']
line_list=['solid','dashed']
marker_list=['o','s']

for i in range(len(input_list)):
    data=np.loadtxt('randomBatching/mu_muadj_std_stdadj_epsilon_'+input_list[i]+'SNPs.dat')

    bs=data.T[0]
    mu=data.T[1]
    mu_adj=data.T[2]
    sigma=data.T[3]
    sigma_adj=data.T[4]

    plt.plot(bs,mu,linewidth=2.5,color=color_list[i],linestyle='solid',marker=marker_list[i],markersize=0 ,label='Random Global QNN, mu '+input_list[i]+' synthetic SNPs')
    plt.plot(bs,mu_adj,linewidth=2.5,color='red',linestyle='solid',marker=marker_list[i],markersize=0 ,label='Random Global QNN, mu_adj '+input_list[i]+' synthetic SNPs')
    plt.plot(bs,sigma,linewidth=2.5,color='blue',linestyle='None',marker=marker_list[i],markersize=2 ,label='Random Global QNN, sigma '+input_list[i]+' synthetic SNPs')
    plt.plot(bs,sigma_adj,linewidth=2.5,color='blue',linestyle='None',marker=marker_list[i],markersize=2 ,label='Random Global QNN, sigma_adj '+input_list[i]+' synthetic SNPs')


#for i in range(len(input_list)):
#    data=np.loadtxt('smartBatching/smartBatching_mu_muadj_std_stdadj_epsilon_'+input_list[i]+'SNPs.dat')

#    bs_smart=data.T[0]
#    bs_smart=651./2/np.array(bs_smart)
#    epsilon_smart=data.T[-1]

#    plt.plot(bs_smart,epsilon_smart,linewidth=2.5,color=color_list_2[i],linestyle=line_list[i], label='Smart Global QNN, '+input_list[i]+' synthetic SNPs',marker=marker_list[i], markersize=8,)

plt.legend(loc='upper right',
          framealpha=0.9, fontsize=12)
plt.show()