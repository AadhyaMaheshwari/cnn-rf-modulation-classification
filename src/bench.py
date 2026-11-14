from net import *
X,Y,S,sp=load(); n=Net(0); rng=np.random.default_rng(0)
t=time.time()
for i in range(5):
    b=np.arange(i*128,(i+1)*128); z,c=n.fwd(X[b],True,rng); l,G=n.bwd(z,Y[b],c); n.step(G,1e-3)
print((time.time()-t)/5,'s/step',l)
# gradient check on W1 and W2
n=Net(1); b=np.arange(64); z,c=n.fwd(X[b],False); l,G=n.bwd(z,Y[b],c)
for k,idx in [('W1',(1,5)),('W2',(100,3)),('W3',(500,7)),('b1',(2,))]:
    e=1e-2; n.P[k][idx]+=e; l1=n.bwd(*( (lambda zc:(zc[0],Y[b],zc[1]))(n.fwd(X[b],False)) ))[0]; n.P[k][idx]-=2*e
    l2=n.bwd(*( (lambda zc:(zc[0],Y[b],zc[1]))(n.fwd(X[b],False)) ))[0]; n.P[k][idx]+=e
    print(k,(l1-l2)/(2*e),G[k][idx])
