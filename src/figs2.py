import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9})
R=json.load(open('results/results.json')); R.update(json.load(open('results/results2.json'))); SN=[-20,-16,-12,-8,-4,0,4,8,12,16,20,24,28,30]
MODS=['QPSK','BPSK','QAM16','QAM64','8PSK','PAM4','GFSK','CPFSK']
NAVY='#1F3864'; OR='#E07B39'; GR='#7f7f7f'; GN='#4c9a52'; RD='#c0392b'
# fig3
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
sty={'FP32':('k','--','float32'),'QAT_16w16a':(NAVY,'-','QAT 16w16a'),'QAT_8w16a':(OR,'-','QAT 8w16a'),'QAT_4w16a':(RD,'-','QAT 4w16a')}
for k,(c,l,n) in sty.items(): ax[0].plot(SN,np.array(R[k]['snr_acc'])*100,l,color=c,marker='o',ms=3,label=n)
ax[0].set_xlabel('SNR (dB)'); ax[0].set_ylabel('test accuracy (%)'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title('(a) QAT models',fontsize=9)
sty2={'FP32':('k','--','float32'),'PTQ_8w16a':(OR,'-','PTQ 8w16a'),'PTQ_4w16a':(RD,'-','PTQ 4w16a'),'QAT_4w16a':(RD,':','QAT 4w16a')}
for k,(c,l,n) in sty2.items(): ax[1].plot(SN,np.array(R[k]['snr_acc'])*100,l,color=c,marker='o',ms=3,label=n)
ax[1].set_xlabel('SNR (dB)'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('(b) PTQ against QAT',fontsize=9)
for a in ax: a.spines[['top','right']].set_visible(False); a.grid(alpha=.25)
plt.tight_layout(); plt.savefig('figs/fig3.png',dpi=200); plt.close()
# fig4 training curves
fig,ax=plt.subplots(1,2,figsize=(7.2,2.5))
h=R['FP32']['hist']; ax[0].plot([x[0] for x in h],color=NAVY); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('training loss'); ax[0].set_title('(a) float32 training loss',fontsize=9)
ax2=ax[0].twinx(); ax2.plot([x[1]*100 for x in h],color=OR); ax2.set_ylabel('validation accuracy (%)',color=OR)
for k,c in [('QAT_16w16a',NAVY),('QAT_8w16a',OR),('QAT_4w16a',RD)]:
    ax[1].plot([x[1]*100 for x in R[k]['hist']],color=c,marker='o',ms=3,label=k.replace('QAT_',''))
ax[1].set_xlabel('QAT epoch'); ax[1].set_ylabel('validation accuracy (%)'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('(b) QAT fine-tuning',fontsize=9)
for a in ax: a.spines[['top']].set_visible(False)
plt.tight_layout(); plt.savefig('figs/fig4.png',dpi=200); plt.close()
# fig5 confusion matrix
cm=np.array(R['QAT_16w16a']['cm_hi'],float); cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(4.2,3.6)); im=ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(8),MODS,rotation=45,ha='right',fontsize=7); ax.set_yticks(range(8),MODS,fontsize=7)
ax.set_xlabel('predicted'); ax.set_ylabel('true')
for i in range(8):
    for j in range(8): ax.text(j,i,f'{cmn[i,j]*100:.0f}',ha='center',va='center',fontsize=6,color='white' if cmn[i,j]>.5 else 'black')
plt.colorbar(im,fraction=.046); plt.tight_layout(); plt.savefig('figs/fig5.png',dpi=200); plt.close()
# fig6 accuracy vs storage
W=dict(W1=192,W2=6144,W3=253952,W4=1024); K=['W1','W2','W3','W4']
def kib(b): return sum(W[k]*bb for k,bb in zip(K,b))/8/1024
cfg={'QAT_16w16a':(16,)*4,'QAT_8w16a':(8,)*4,'QAT_4w16a':(4,)*4,'QAT_2w16a':(2,)*4,'MP_c8_d4':(8,8,4,4),'MP_c16_d4':(16,16,4,4),'MP_c4_d16':(4,4,16,16),'MP_c4_d2':(4,4,2,2),'MP_c8_d2':(8,8,2,2),'CTRL_FP32_ft':(32,)*4}
lab={'QAT_16w16a':'16/16/16/16','QAT_8w16a':'8/8/8/8','QAT_4w16a':'4/4/4/4','QAT_2w16a':'2/2/2/2','MP_c8_d4':'8/8/4/4','MP_c16_d4':'16/16/4/4','MP_c4_d16':'4/4/16/16','MP_c4_d2':'4/4/2/2','MP_c8_d2':'8/8/2/2','CTRL_FP32_ft':'float32'}
fig,ax=plt.subplots(figsize=(6.0,3.2))
for k,b in cfg.items():
    c=OR if k.startswith('MP') else (GR if k.startswith('CTRL') else NAVY)
    x=kib(b); y=R[k]['hi20']*100; ax.scatter(x,y,color=c,s=26); ax.annotate(lab[k],(x,y),textcoords='offset points',xytext=(4,4),fontsize=6.5)
ax.set_xscale('log'); ax.set_xlabel('packed weight storage (KiB, log scale)'); ax.set_ylabel('mean accuracy, SNR >= 20 dB (%)')
ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig('figs/fig6.png',dpi=200); plt.close()
# fig7 per-layer PTQ sensitivity
fig,ax=plt.subplots(figsize=(5.2,2.6)); L=['conv1','conv2','dense1','dense2']
for j,(b,c) in enumerate([(4,NAVY),(3,OR),(2,RD)]):
    v=[R[f'PTQ_only_{k}_{b}bit' if b!=4 else f'PTQ_only_{k}_4bit']['hi20']*100 for k in K]
    ax.bar(np.arange(4)+(j-1)*0.26,v,0.26,color=c,label=f'{b}-bit')
ax.axhline(R['FP32']['hi20']*100,color='k',ls='--',lw=1); ax.text(3.45,R['FP32']['hi20']*100+1.2,'float32',fontsize=7,ha='right')
ax.set_xticks(range(4),L); ax.set_ylabel('accuracy, SNR >= 20 dB (%)'); ax.legend(frameon=False,fontsize=7,ncol=3,loc='lower left'); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figs/fig7.png',dpi=200); plt.close(); print('figs ok')
