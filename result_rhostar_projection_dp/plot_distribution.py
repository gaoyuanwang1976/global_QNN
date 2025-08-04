import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1, 2, figsize=(6.4, 4.8))
ranges=(0.3,0.6)
resolution=0.01
range_size=ranges[1]-ranges[0]
num_bin=int(range_size/resolution)
input_list=['32','162']

data_list=[]
for i in range(len(input_list)):
    data_list.append(np.loadtxt('randomBatching/128synSNPs_distribution/'+input_list[i]+'synSNPs_withinDataset_LoopOverEachState_rhostar_projection_prob.dat'))
#data_list.append(np.loadtxt(input_list[1]+'synSNPs_withinDataset_LoopOverEachState_rhostar_projection_prob.dat'))
#data_list.append(np.loadtxt(input_list[2]+'synSNPs_withinDataset_LoopOverEachState_rhostar_projection_prob.dat'))


from scipy.stats import norm
mu_list=[]
std_list=[]
mu_adj_list=[]
std_adj_list=[]
for i in range(len(input_list)):
    mu_list.append(norm.fit(data_list[i][0])[0])
    std_list.append(norm.fit(data_list[i][0])[1])
    mu_adj_list.append(norm.fit(data_list[i][1])[0])
    std_adj_list.append(norm.fit(data_list[i][1])[1])
    
color_list=['cyan','orange','green']
adj_list=['blue','red','mediumseagreen']
'''
histo1_all=np.histogram(data1[0], bins=num_bin, range=ranges,density=True)
histo1_adj=np.histogram(data1[1], bins=num_bin, range=ranges,density=True)
histo2_all=np.histogram(data2[0], bins=num_bin, range=ranges,density=True)
histo2_adj=np.histogram(data2[1], bins=num_bin, range=ranges,density=True)
histo3_all=np.histogram(data3[0], bins=num_bin, range=ranges,density=True)
histo3_adj=np.histogram(data3[1], bins=num_bin, range=ranges,density=True)




plt.plot(histo1_all[1][:-1]+resolution,histo1_all[0],label='batch size 7, global state',linestyle='solid',color='steelblue',linewidth=2)
plt.plot(histo1_adj[1][:-1]+resolution,histo1_adj[0],label='batch size 7, adjacent state',linestyle='dashed',color='steelblue',linewidth=2)
plt.plot(histo2_all[1][:-1]+resolution,histo2_all[0],label='batch size 64, global state',linestyle='solid',color='indianred',linewidth=2)
plt.plot(histo2_adj[1][:-1]+resolution,histo2_adj[0],label='batch size 64, adjacent state',linestyle='dashed',color='indianred',linewidth=2)
plt.plot(histo3_all[1][:-1]+resolution,histo3_all[0],label='batch size 324, global state',linestyle='solid',color='mediumseagreen',linewidth=2)
plt.plot(histo3_adj[1][:-1]+resolution,histo3_adj[0],label='batch size 324, adjacent state',linestyle='dashed',color='mediumseagreen',linewidth=2)
'''

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
bins=np.arange(0.3,0.6,0.005)
for i in range(len(input_list)):
    ax[i].set_title('batch size '+input_list[i])
    pdf = norm.pdf(x, mu_list[i], std_list[i])
    ax[i].plot(x, pdf, 'r--', label='global state, fitted',color=color_list[i])
    pdf_adj = norm.pdf(x, mu_adj_list[i], std_adj_list[i])
    
    ax[i].plot(x, pdf_adj, 'r--', label='adjacent state, fitted',color=adj_list[i])
    ax[i].hist(data_list[i][0],bins=bins,alpha=0.3,density=True,label='global state',color=color_list[i])
    ax[i].hist(data_list[i][1],bins=bins,alpha=0.3,density=True,label='adjacent state',color=adj_list[i])
    ax[i].legend()
    ax[i].axis([0.3, 0.6, 0, 25])
    ax[i].tick_params(labelsize=12)
    #ax[i].tick_params(labelsize=14)

fig.supxlabel('$ρ^*$ projection outcome', fontsize=12,fontweight='bold')
fig.supylabel('Probability', fontsize=12,fontweight='bold')
#ax[0].xlabel('$ρ^*$ projection outcome', fontsize=12, fontweight='bold')
#ax[0].ylabel('Probability', fontsize=12, fontweight='bold')

fig.suptitle("Measurement outcome - 128 synthetic SNPs", fontsize=14,fontweight='bold')


#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.axis([0.3, 0.6, 0, 20])
#plt.legend(loc='upper right',
#          framealpha=0.9, fontsize=12)
plt.show()