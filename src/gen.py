import numpy as np, sys, time
MODS=['QPSK','BPSK','QAM16','QAM64','8PSK','PAM4','GFSK','CPFSK']
SNRS=[-20,-16,-12,-8,-4,0,4,8,12,16,20,24,28,30]
SPS=8; NL=5120; CROP=4096; W=128
def rc_taps(alpha=0.5, span=10, sps=SPS):
    t=np.arange(-(span*sps)//2,(span*sps)//2+1)/sps
    den=1-(2*alpha*t)**2
    h=np.sinc(t)*np.cos(np.pi*alpha*t)/np.where(np.abs(den)<1e-9,1,den)
    h[np.abs(den)<1e-9]=np.pi/4*np.sinc(1/(2*alpha))
    return h
RC=rc_taps()
def qam(M):
    m=int(np.sqrt(M)); a=np.arange(-(m-1),m,2)
    return (a[:,None]+1j*a[None,:]).ravel()
CONS={'BPSK':np.array([-1,1],complex),
 'QPSK':np.exp(1j*(np.pi/4+np.pi/2*np.arange(4))),
 '8PSK':np.exp(1j*2*np.pi*np.arange(8)/8),
 'QAM16':qam(16),'QAM64':qam(64),'PAM4':np.array([-3,-1,1,3],complex)}
def gauss_pulse(bt=0.35,span=4,sps=SPS):
    t=np.arange(-span*sps//2,span*sps//2+1)/sps
    s=np.sqrt(np.log(2))/(2*np.pi*bt)
    g=np.exp(-t**2/(2*s**2)); g/=g.sum()
    return np.convolve(np.ones(sps),g)
def fsk(bits,h,pulse):
    a=2*bits-1.0
    up=np.zeros(len(a)*SPS); up[::SPS]=a
    f=np.convolve(up,pulse)[:len(a)*SPS+len(pulse)-1]
    f=f[(len(pulse)-1)//2:(len(pulse)-1)//2+len(a)*SPS]
    ph=2*np.pi*np.cumsum(h/(2*SPS)*f)
    return np.exp(1j*ph)
GP=gauss_pulse(); RP=np.ones(SPS)
def make(mod,rng):
    n=NL//SPS
    if mod in CONS:
        s=CONS[mod][rng.integers(0,len(CONS[mod]),n)]
        up=np.zeros(NL,complex); up[::SPS]=s
        x=np.convolve(up,RC,mode='same')
    elif mod=='GFSK': x=fsk(rng.integers(0,2,n),1.0,GP)
    else: x=fsk(rng.integers(0,2,n),0.5,RP)
    return x/np.sqrt(np.mean(np.abs(x)**2))
DEL=np.array([0,1.8,3.4]); PG=10**(np.array([0,-2,-10])/20)
def channel(x,rng,K=4.0):
    cn=lambda: (rng.standard_normal()+1j*rng.standard_normal())/np.sqrt(2)
    g=[np.sqrt(K/(K+1))*np.exp(2j*np.pi*rng.random())+np.sqrt(1/(K+1))*cn(), cn(), cn()]
    g=np.array(g)*PG
    f=np.fft.fftfreq(len(x))
    H=sum(gi*np.exp(-2j*np.pi*f*d) for gi,d in zip(g,DEL))
    return np.fft.ifft(np.fft.fft(x)*H)
def one(mod,snr,rng):
    y=channel(make(mod,rng),rng)
    off=rng.integers(200,NL-CROP-200)
    y=y[off:off+CROP]
    d=rng.uniform(-5e3,5e3)/4e6
    y=y*np.exp(2j*np.pi*(d*np.arange(CROP)+rng.random()))
    ps=np.mean(np.abs(y)**2); sg=np.sqrt(ps/10**(snr/10)/2)
    y=y+sg*(rng.standard_normal(CROP)+1j*rng.standard_normal(CROP))
    w=y.reshape(32,W)[::4]                      # 8 spaced 128-sample windows
    w=w/np.sqrt(np.mean(np.abs(w)**2,axis=1,keepdims=True))
    return np.stack([w.real,w.imag],1).astype(np.float32)   # (8,2,128)
if __name__=='__main__':
    NLONG=60; rng=np.random.default_rng(2026); t=time.time()
    X=[];Y=[];S=[];G=[]
    for mi,m in enumerate(MODS):
        for si,snr in enumerate(SNRS):
            for l in range(NLONG):
                w=one(m,snr,rng); X.append(w); Y+= [mi]*8; S+=[si]*8; G+=[l]*8
    X=np.concatenate(X); Y=np.array(Y); S=np.array(S); G=np.array(G)
    split=np.where(G<40,0,np.where(G<50,1,2))      # grouped by long frame: 40/10/10
    np.savez('data.npz',X=X,Y=Y,S=S,split=split)
    print(X.shape,[int((split==k).sum()) for k in range(3)],time.time()-t)
