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
plt.title('Privacy - 256 PsychENCODE SNPs', 
          fontsize=14, fontweight='bold', pad=20)

input_list=['256']
color_list=['#e74c3c']
color_list_2=['#3498db']
line_list=['solid','dashed']
marker_list=['o','s']

for i in range(len(input_list)):
    data=np.loadtxt('PsychEncode_mu_muadj_std_stdadj_epsilon_'+input_list[i]+'SNPs.dat')

    bs=data.T[0]
    epsilon=data.T[-1]

    plt.plot(bs,epsilon,linewidth=2.5,color=color_list[i],linestyle=line_list[i],marker=marker_list[i],markersize=8 ,label='Random Global QNN '+input_list[i]+' SNPs')


for i in range(len(input_list)):
    data=np.loadtxt('PsychEncode_smart_mu_muadj_std_stdadj_epsilon_'+input_list[i]+'SNPs.dat')

    bs_smart=data.T[0]
    bs_smart=651./2/np.array(bs_smart)
    epsilon_smart=data.T[-1]

    plt.plot(bs_smart,epsilon_smart,linewidth=2.5,color=color_list_2[i],linestyle=line_list[i], label='Smart Global QNN '+input_list[i]+' SNPs',marker=marker_list[i], markersize=8,)

plt.legend(loc='upper right',
          framealpha=0.9, fontsize=12)
plt.show()