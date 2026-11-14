from net import *
X,Y,S,sp=load(); tr,va,te=[sp==k for k in range(3)]
Xt,Yt,Xv,Yv,Xe,Ye,Se=X[tr],Y[tr],X[va],Y[va],X[te],Y[te],S[te]
R={}
def ev(net,name,hist=None):
    p=net.predict(Xe); a=acc_by_snr(p,Ye,Se)
    cm=np.zeros((8,8),int)
    h=Se>=10
    for t,q in zip(Ye[h],p[h]): cm[t,q]+=1
    R[name]={'snr_acc':a,'overall':float((p==Ye).mean()),'hi20':float(np.mean(a[10:])),'ge4':float(np.mean(a[6:])),
             'cm_hi':cm.tolist(),'hist':hist}
    print('EVAL',name,round(R[name]['overall'],4),round(R[name]['hi20'],4),flush=True)
    json.dump(R,open('results/results.json','w'))
base=Net(0); h=train(base,Xt,Yt,Xv,Yv,25,1e-3,0,'FP32'); ev(base,'FP32',h)
import copy
P0={k:v.copy() for k,v in base.P.items()}; np.savez('checkpoints/fp32_ckpt.npz',**P0)
def clone(bits,abits):
    n=Net(0); n.P={k:v.copy() for k,v in P0.items()}; n.bits=dict(bits); n.abits=abits
    n.m={k:np.zeros_like(v) for k,v in n.P.items()}; n.v={k:np.zeros_like(v) for k,v in n.P.items()}; n.t=0; return n
# post-training quantisation (no retraining)
for b in (16,8,4):
    n=clone({k:b for k in ['W1','W2','W3','W4']},16); ev(n,f'PTQ_{b}w16a')
for k in ['W1','W2','W3','W4']:
    bits={q:32 for q in ['W1','W2','W3','W4']}; bits[k]=4
    n=clone(bits,32); ev(n,f'PTQ_only_{k}_4bit')
cfg={'QAT_16w16a':(16,16,16,16),'QAT_8w16a':(8,8,8,8),'QAT_4w16a':(4,4,4,4),
     'MP_c8_d4':(8,8,4,4),'MP_c16_d4':(16,16,4,4),'MP_c4_d16':(4,4,16,16)}
for name,b in cfg.items():
    bits=dict(zip(['W1','W2','W3','W4'],b)); n=clone(bits,16)
    # calibrate activation ranges
    n.fwd(Xt[:512],True,np.random.default_rng(0),0)
    h=train(n,Xt,Yt,Xv,Yv,10,3e-4,1,name); ev(n,name,h)
print('DONE')
