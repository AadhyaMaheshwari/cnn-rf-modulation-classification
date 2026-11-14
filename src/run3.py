import sys
from net import *
X,Y,S,sp=load(); tr,va,te=[sp==k for k in range(3)]
Xt,Yt,Xv,Yv,Xe,Ye,Se=X[tr],Y[tr],X[va],Y[va],X[te],Y[te],S[te]
K=['W1','W2','W3','W4']; P0=dict(np.load('checkpoints/fp32_ckpt.npz')); R=json.load(open('results/results2.json'))
cfg={'QAT_2w16a':(2,2,2,2),'MP_c4_d2':(4,4,2,2),'MP_c8_d2':(8,8,2,2)}
name=sys.argv[1]; b=cfg[name]
n=Net(0); n.P={k:v.copy() for k,v in P0.items()}; n.bits=dict(zip(K,b)); n.abits=16
n.fwd(Xt[:512],True,np.random.default_rng(0),0)
h=train(n,Xt,Yt,Xv,Yv,10,3e-4,1,name,patience=10)
p=n.predict(Xe); a=acc_by_snr(p,Ye,Se)
R[name]={'snr_acc':a,'overall':float((p==Ye).mean()),'hi20':float(np.mean(a[10:])),'ge4':float(np.mean(a[6:])),'hist':h}
json.dump(R,open('results/results2.json','w')); print('EVAL',name,R[name]['overall'],R[name]['hi20'])
