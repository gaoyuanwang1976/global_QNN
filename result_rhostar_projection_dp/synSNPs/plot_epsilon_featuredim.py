import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')

#plt.axis([0, 550, 0, 10])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Privacy Parameter ε', fontsize=12, fontweight='bold')
plt.xlabel('Number of SNPs', fontsize=12, fontweight='bold')
plt.title('Privacy - Various datasets', 
          fontsize=14, fontweight='bold', pad=20)

input_list=['64','32']
bs_smart=['65.1','32.55']
input_list_smart=['5','10']
color_list=['salmon','#e74c3c']
color_list_2=['cyan','#3498db']
line_list=['solid','dashed']
marker_list=['o','s']

for i in range(len(input_list)):
    data=np.loadtxt('randomBatching/mu_muadj_std_stdadj_epsilon_BS'+input_list[i]+'.dat')

    snps=data.T[0]
    epsilon=data.T[-1]

    plt.plot(snps,epsilon,linewidth=2.5,color=color_list[i],linestyle=line_list[i],marker=marker_list[i],markersize=8 ,label='Random Global QNN, batch size '+input_list[i])
    #plt.plot(bs,epsilon,linewidth=2.5,color=color_list[i],linestyle=line_list[i],marker=marker_list[i],markersize=8 ,label='Random Global QNN, batch size'+input_list[i])
    

for i in range(len(input_list)):
    data=np.loadtxt('smartBatching/smartBatching_mu_muadj_std_stdadj_epsilon_BN'+input_list_smart[i]+'.dat')
    snps=data.T[0]
    epsilon_smart=data.T[-1]

    plt.plot(snps,epsilon_smart,linewidth=2.5,color=color_list_2[i],linestyle=line_list[i], label='Smart Global QNN, batch size '+bs_smart[i],marker=marker_list[i], markersize=8,)

plt.legend(loc='upper left',
          framealpha=0.9, fontsize=12)
plt.show()