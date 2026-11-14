import json
L={'conv1':dict(w=64*1*1*3,b=64,mac=64*3*252,out=64*2*126),
   'conv2':dict(w=16*64*2*3,b=16,mac=124*384*16,out=16*124),
   'dense1':dict(w=1984*128,b=128,mac=1984*128,out=128),
   'dense2':dict(w=128*8,b=8,mac=128*8,out=8)}
W=sum(v['w'] for v in L.values()); B=sum(v['b'] for v in L.values()); M=sum(v['mac'] for v in L.values())
print(L); print('weights',W,'biases',B,'total',W+B,'MACs',M)
for k,v in L.items(): print(k,'wshare %.2f%%'%(100*v['w']/W),'macshare %.2f%%'%(100*v['mac']/M))
# Table 3 consistency
T3={'16w16a':dict(lut=[3245,14822,7826,754],tl=26976,reg=[4268,20187,14282,789],tr=39791,bram=[0,32,136,0],tb=169,dsp=[64,256,128,8],td=456),
    '8w16a':dict(lut=[4588,11718,6921,564],tl=24045,reg=[5374,15931,13372,659],tr=35602,bram=[0,32,72,0],tb=105,dsp=[64,256,128,8],td=456),
    '4w16a':dict(lut=[4233,10436,6531,476],tl=21930,reg=[4916,14034,12988,599],tr=32803,bram=[0,32,72,0],tb=105,dsp=[64,256,128,8],td=456)}
for k,v in T3.items():
    print(k,'LUT sum',sum(v['lut']),'vs',v['tl'],'| REG',sum(v['reg']),'vs',v['tr'],'| BRAM',sum(v['bram']),'vs',v['tb'],'| DSP',sum(v['dsp']),'vs',v['td'])
print('DSP util',456/4272,'BRAM util 169',169/1080,'105',105/1080)
print('LUT vs16: 8w %.1f%% 4w %.1f%%'%(100*(24045/26976-1),100*(21930/26976-1)),'BRAM 8w %.1f%%'%(100*(105/169-1)))
print('dense1 BRAM share 16w %.1f%%'%(100*136/169))
# BRAM lower bound for dense1 weights
for b,u in [(16,136),(8,72),(4,72)]:
    bits=253952*b; print('dense1',b,'bit',bits,'min BRAM36',bits/36864,'reported',u)
# weight memory per precision
for b in (16,8,4): print(b,'bit weights KiB',W*b/8/1024)
# cycles at 128 MHz, frame=128 complex samples at 4 Msps
print('frame budget cycles',128*32)
cyc={'conv1':252*3,'conv2':124*384*16//256,'dense1':1984*128//128,'dense2':128*8//8}
print(cyc,sum(cyc.values()),'max',max(cyc.values()),'util',max(cyc.values())/4096)
print('MACs/s at 31250 frames/s:',M*31250/1e9,'GMAC/s ; frame time us',128/4)
print('channel: delays ns',4.2/3e8*1e9,7.97/3e8*1e9,'samples@128MHz',4.2/3e8*128e6,7.97/3e8*128e6,'doppler Hz',1.7*700e6/3e8)
print('dataset frames 8*330000=',8*330000,' expected 1000*14*32=',1000*14*32,'ratio',330000/(1000*14*32))
print('16-bit vs float', 80-76)
