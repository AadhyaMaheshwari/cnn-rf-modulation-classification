import numpy as np, json, time, sys
from numpy.lib.stride_tricks import sliding_window_view as swv
F=np.float32
def fq(W,b):
    if b>=32: return W
    q=2**(b-1)-1; s=np.abs(W).max()/q
    return (np.clip(np.round(W/s),-q,q)*s).astype(F)
class Net:
    # conv1 64x(1x3) -> conv2 16x(2x3 over 64ch) -> dense 1984x128 -> dense 128x8
    def __init__(s,seed=0):
        r=np.random.default_rng(seed)
        he=lambda fi,sh: (r.standard_normal(sh)*np.sqrt(2/fi)).astype(F)
        s.P={'W1':he(3,(3,64)),'b1':np.zeros(64,F),'W2':he(384,(384,16)),'b2':np.zeros(16,F),
             'W3':he(1984,(1984,128)),'b3':np.zeros(128,F),'W4':he(128,(128,8)),'b4':np.zeros(8,F)}
        s.bits={'W1':32,'W2':32,'W3':32,'W4':32}; s.abits=32; s.am={}
        s.m={k:np.zeros_like(v) for k,v in s.P.items()}; s.v={k:np.zeros_like(v) for k,v in s.P.items()}; s.t=0
    def aq(s,a,name,train):
        if s.abits>=32: return a
        mx=float(np.abs(a).max())
        if train: s.am[name]=0.99*s.am.get(name,mx)+0.01*mx if name in s.am else mx
        sc=max(s.am.get(name,mx),1e-6)/(2**(s.abits-1)-1)
        return (np.clip(np.round(a/sc),-(2**(s.abits-1)),2**(s.abits-1)-1)*sc).astype(F)
    def fwd(s,x,train=False,rng=None,drop=0.3):
        P=s.P; Wq={k:fq(P[k],s.bits[k]) for k in s.bits}; B=x.shape[0]
        x=s.aq(x,'x',train)
        xc=swv(x,3,axis=2).reshape(B*252,3)
        z1=xc@Wq['W1']+P['b1']; a1=s.aq(np.maximum(z1,0),'a1',train).reshape(B,2,126,64)
        cols=swv(a1,3,axis=2).transpose(0,2,1,4,3).reshape(B*124,384)
        z2=cols@Wq['W2']+P['b2']; a2=s.aq(np.maximum(z2,0),'a2',train).reshape(B,1984)
        z3=a2@Wq['W3']+P['b3']; a3=s.aq(np.maximum(z3,0),'a3',train)
        mask=None
        if train and drop>0:
            mask=((rng.random(a3.shape)>drop)/(1-drop)).astype(F); a3=a3*mask
        z4=a3@Wq['W4']+P['b4']
        c=(xc,z1,cols,z2,a2,z3,a3,mask,Wq,B)
        return z4,c
    def bwd(s,z4,y,c):
        xc,z1,cols,z2,a2,z3,a3,mask,Wq,B=c
        p=np.exp(z4-z4.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        loss=-np.log(p[np.arange(B),y]+1e-12).mean()
        d=p; d[np.arange(B),y]-=1; d/=B; G={}
        G['W4']=a3.T@d; G['b4']=d.sum(0); da3=d@Wq['W4'].T
        if mask is not None: da3=da3*mask
        dz3=da3*(z3>0); G['W3']=a2.T@dz3; G['b3']=dz3.sum(0); da2=dz3@Wq['W3'].T
        dz2=da2.reshape(-1,16)*(z2>0); G['W2']=cols.T@dz2; G['b2']=dz2.sum(0)
        dc=(dz2@Wq['W2'].T).reshape(B,124,2,3,64); da1=np.zeros((B,2,126,64),F)
        for k in range(3): da1[:,:,k:k+124,:]+=dc[:,:,:,k,:].transpose(0,2,1,3)
        dz1=(da1*(z1.reshape(B,2,126,64)>0)).reshape(B*252,64)
        G['W1']=xc.T@dz1; G['b1']=dz1.sum(0)
        return loss,G
    def step(s,G,lr,wd=0):
        s.t+=1
        for k,g in G.items():
            s.m[k]=0.9*s.m[k]+0.1*g; s.v[k]=0.999*s.v[k]+0.001*g*g
            mh=s.m[k]/(1-0.9**s.t); vh=s.v[k]/(1-0.999**s.t)
            s.P[k]-=(lr*mh/(np.sqrt(vh)+1e-8)).astype(F)
    def predict(s,X,bs=512):
        out=[]
        for i in range(0,len(X),bs): out.append(s.fwd(X[i:i+bs])[0].argmax(1))
        return np.concatenate(out)
def load():
    d=np.load('data.npz'); return d['X'],d['Y'],d['S'],d['split']
def acc_by_snr(pred,Y,S):
    return [float((pred[S==i]==Y[S==i]).mean()) for i in range(14)]
def train(net,X,Y,Xv,Yv,epochs,lr,seed,log,bs=128,patience=8):
    rng=np.random.default_rng(seed); best=(-1,None,None); bad=0; hist=[]
    for ep in range(epochs):
        t=time.time(); idx=rng.permutation(len(X)); tl=0;n=0
        for i in range(0,len(idx)-bs+1,bs):
            b=idx[i:i+bs]; z,c=net.fwd(X[b],True,rng); l,G=net.bwd(z,Y[b],c); net.step(G,lr); tl+=l;n+=1
        va=float((net.predict(Xv)==Yv).mean()); hist.append((float(tl/n),float(va)))
        print(log,ep,round(tl/n,4),round(va,4),round(time.time()-t,1),flush=True)
        if va>best[0]: best=(va,{k:v.copy() for k,v in net.P.items()},dict(net.am)); bad=0
        else:
            bad+=1
            if bad>=patience: break
    net.P=best[1]; net.am=best[2]; return hist
