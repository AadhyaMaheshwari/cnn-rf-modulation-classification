import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.family':'serif','font.size':9})
NAVY='#1F3864'; OR='#E07B39'; GR='#7f7f7f'
# Fig 1: weight share vs MAC share
L=['conv1','conv2','dense1','dense2']; w=[192,6144,253952,1024]; m=[48384,761856,253952,1024]
ws=np.array(w)/sum(w)*100; ms=np.array(m)/sum(m)*100
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
x=np.arange(4); ax[0].bar(x-0.2,ws,0.4,color=NAVY,label='weights'); ax[0].bar(x+0.2,ms,0.4,color=OR,label='MACs')
ax[0].set_xticks(x,L); ax[0].set_ylabel('share of network total (%)'); ax[0].legend(frameon=False); ax[0].set_title('(a) Weights and MACs per layer',fontsize=9)
for i,(a,b) in enumerate(zip(ws,ms)): ax[0].text(i-0.2,a+1.5,f'{a:.1f}',ha='center',fontsize=7); ax[0].text(i+0.2,b+1.5,f'{b:.1f}',ha='center',fontsize=7)
cyc=[756,2976,1984,128]; ax[1].bar(x,cyc,color=GR); ax[1].axhline(4096,color='crimson',ls='--',lw=1); ax[1].text(3.4,4150,'frame budget 4,096',ha='right',fontsize=7,color='crimson')
ax[1].set_xticks(x,L); ax[1].set_ylabel('cycles per frame at 128 MHz'); ax[1].set_ylim(0,4700); ax[1].set_title('(b) Cycles per frame (derived)',fontsize=9)
for i,c in enumerate(cyc): ax[1].text(i,c+70,f'{c:,}',ha='center',fontsize=7)
for a in ax: a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figs/fig1.png',dpi=200); plt.close()
# Fig 2: BRAM reported vs lower bound
bits=[16,8,4]; rep=[136,72,72]; lb=[110.2,55.1,27.6]
fig,ax=plt.subplots(figsize=(4.6,2.5)); x=np.arange(3)
ax.bar(x-0.2,rep,0.4,color=NAVY,label='reported (dense layer 1, [1])'); ax.bar(x+0.2,lb,0.4,color=OR,label='bit-packed lower bound')
ax.set_xticks(x,['16-bit','8-bit','4-bit']); ax.set_ylabel('36 Kb block RAMs'); ax.legend(frameon=False,fontsize=7)
for i,(a,b) in enumerate(zip(rep,lb)): ax.text(i-0.2,a+2,str(a),ha='center',fontsize=7); ax.text(i+0.2,b+2,f'{b:.0f}',ha='center',fontsize=7)
ax.spines[['top','right']].set_visible(False); plt.tight_layout(); plt.savefig('figs/fig2.png',dpi=200); plt.close()
