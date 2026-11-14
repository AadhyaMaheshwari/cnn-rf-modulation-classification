from net import *
X,Y,S,sp=load(); tr,va,te=[sp==k for k in range(3)]
Xt,Yt,Xv,Yv,Xe,Ye,Se=X[tr],Y[tr],X[va],Y[va],X[te],Y[te],S[te]
R={}; K=['W1','W2','W3','W4']
P0=dict(np.load('checkpoints/fp32_ckpt.npz'))
def ev(net,name,hist=None):
    p=net.predict(Xe); a=acc_by_snr(p,Ye,Se)
    R[name]={'snr_acc':a,'overall':float((p==Ye).mean()),'hi20':float(np.mean(a[10:])),'ge4':float(np.mean(a[6:])),'hist':hist}
    print('EVAL',name,round(R[name]['overall'],4),round(R[name]['hi20'],4),flush=True); json.dump(R,open('results/results2.json','w'))
def clone(bits,abits):
    n=Net(0); n.P={k:v.copy() for k,v in P0.items()}; n.bits=dict(bits); n.abits=abits; return n
for b in (3,2):
    ev(clone({k:b for k in K},32),f'PTQ_{b}bit_all')
    for k in K:
        bits={q:32 for q in K}; bits[k]=b; ev(clone(bits,32),f'PTQ_only_{k}_{b}bit')
cfg={'CTRL_FP32_ft':((32,)*4,32)}   # 2-bit QAT variants are run with run3.py
for name,(b,ab) in cfg.items():
    n=clone(dict(zip(K,b)),ab)
    if ab<32: n.fwd(Xt[:512],True,np.random.default_rng(0),0)
    h=train(n,Xt,Yt,Xv,Yv,10,3e-4,1,name); ev(n,name,h)
print('DONE')
