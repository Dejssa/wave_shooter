"""Make each cheek of human_female.glb a single flat polygon.

Picks the front-facing cheek (nose side to cheekbone, lip corner to under the
eye), flattens its outline onto a best-fit plane (a few mm at most), and refills
it with coplanar triangles so it reads as one facet. Nearby skin below the eyes
eases along so no folds appear, and the mouth plug corners follow the cheek.
Run on the output of decimate.py.

Usage: python tools/flatten_cheeks.py IN.glb OUT.glb   (needs numpy, pygltflib)
then:  npx @gltf-transform/cli prune OUT.glb OUT.glb
"""
import sys
import numpy as np
from pygltflib import GLTF2, BufferView, Accessor
SRC,DST=sys.argv[1],sys.argv[2]
from pygltflib import GLTF2
DT={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def acc(g,i):
    a=g.accessors[i];bv=g.bufferViews[a.bufferView];blob=g.binary_blob()
    dt=np.dtype(DT[a.componentType]);n=NC[a.type]
    off=(bv.byteOffset or 0)+(a.byteOffset or 0)
    stride=bv.byteStride or dt.itemsize*n
    raw=np.frombuffer(blob,dtype=np.uint8,count=stride*(a.count-1)+dt.itemsize*n,offset=off)
    out=np.lib.stride_tricks.as_strided(raw,shape=(a.count,dt.itemsize*n),strides=(stride,1)).copy().view(dt).reshape(a.count,n)
    return out
g=GLTF2().load(SRC)
joints=[g.nodes[j].name for j in g.skins[0].joints]
prims=[]
for p in g.meshes[0].primitives:
    P=acc(g,p.attributes.POSITION);J=acc(g,p.attributes.JOINTS_0);W=acc(g,p.attributes.WEIGHTS_0)
    prims.append(dict(mat=g.materials[p.material].name,P=P,J=J,W=W,I=acc(g,p.indices).ravel()))
def boneW(pr,name):
    b=joints.index(name);return ((pr['J']==b)*pr['W']).sum(1)
def boundary_loop(T):
    from collections import Counter
    E=Counter(tuple(sorted((t[i],t[(i+1)%3]))) for t in T for i in range(3))
    # directed boundary edges keep the triangles' winding
    nxt={}
    for t in T:
        for i in range(3):
            a,b=t[i],t[(i+1)%3]
            if E[tuple(sorted((a,b)))]==1: nxt[a]=b
    start=next(iter(nxt)); loop=[start]
    while nxt[loop[-1]]!=start: loop.append(nxt[loop[-1]])
    assert len(loop)==len(nxt),'cheek region has holes or several pieces'
    return loop

def earclip(P2):
    idx=list(range(len(P2))); out=[]
    area=sum(P2[i-1,0]*P2[i,1]-P2[i,0]*P2[i-1,1] for i in range(len(P2)))
    s=1 if area>0 else -1
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    while len(idx)>3:
        for k in range(len(idx)):
            i0,i1,i2=idx[k-1],idx[k],idx[(k+1)%len(idx)]
            a,b,c=P2[i0],P2[i1],P2[i2]
            if s*cross(a,b,c)<=1e-12: continue
            if any(s*cross(a,b,P2[j])>0 and s*cross(b,c,P2[j])>0 and s*cross(c,a,P2[j])>0
                   for j in idx if j not in (i0,i1,i2)): continue
            out.append((i0,i1,i2)); idx.pop(k); break
        else: raise RuntimeError('ear clipping failed')
    out.append(tuple(idx)); return out

blob=bytearray(g.binary_blob())
def push(arr,target):
    b=arr.tobytes()
    while len(blob)%4: blob.append(0)
    off=len(blob); blob.extend(b)
    g.bufferViews.append(BufferView(buffer=0,byteOffset=off,byteLength=len(b),target=target))
    return len(g.bufferViews)-1
def A(arr,ctype,typ,target,mm=False):
    a=Accessor(bufferView=push(arr,target),componentType=ctype,count=len(arr),type=typ)
    if mm: a.min=arr.min(0).tolist(); a.max=arr.max(0).tolist()
    g.accessors.append(a); return len(g.accessors)-1

k=max((i for i,p in enumerate(g.meshes[0].primitives) if g.materials[p.material].name=='SkinF'),
      key=lambda i:len(prims[i]['I']))
p=g.meshes[0].primitives[k]; pr=prims[k]
P,I,J,W=pr['P'].astype(np.float64),pr['I'],pr['J'],pr['W']
N0=acc(g,p.attributes.NORMAL); UV=acc(g,p.attributes.TEXCOORD_0) if p.attributes.TEXCOORD_0 is not None else None
_,first,inv=np.unique(np.round(P,5),axis=0,return_index=True,return_inverse=True); inv=inv.ravel()
wP=P[first].copy()
T=inv[I].reshape(-1,3); corner=I.reshape(-1,3)       # welded ids / original corner ids
keep=np.ones(len(T),bool); newtris=[]
for side in (1,-1):
    # front-facing cheek: every corner between the nose side and the cheekbone,
    # the lip corner and the under-eye row (the side of the face turns ~50 deg away)
    X=wP[T][...,0]*side; Y=wP[T][...,1]; Z=wP[T][...,2]
    fn=np.cross(wP[T[:,1]]-wP[T[:,0]],wP[T[:,2]]-wP[T[:,0]])
    sel=((X>0.014)&(X<0.057)&(Y>1.588)&(Y<1.648)&(Z<-0.07)).all(1)&(fn[:,2]<0)
    loop=boundary_loop(T[sel])
    B=wP[loop]
    # best-fit plane of the whole cheek, then flatten its outline onto it
    c0=B.mean(0)
    n=np.linalg.svd(B-c0)[2][2]; n*=-np.sign(n[2]) if n[2]>0 else 1
    d=(B-c0)@n; wP[loop]=B-np.outer(d,n)
    u=np.cross(n,[0,1,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
    P2=np.c_[(wP[loop]-c0)@u,(wP[loop]-c0)@v]
    tris=[(loop[a],loop[b],loop[c]) for a,b,c in earclip(P2)]
    # keep the winding of the removed faces (outward)
    t0=np.array(tris[0]); f0=np.cross(wP[t0[1]]-wP[t0[0]],wP[t0[2]]-wP[t0[0]])
    if f0@n<0: tris=[(a,c,b) for a,b,c in tris]
    keep&=~sel; newtris+=tris
    print(f'side {side:+d}: {sel.sum()} tris -> 1 polygon ({len(loop)} corners, {len(tris)} tris); '
          f'outline moved max {np.abs(d).max()*1000:.1f} mm')

# rebuild the primitive: flat-shaded, per-corner attributes from the welded source vertex
allT=np.vstack([T[keep],np.array(newtris)])
src=np.vstack([corner[keep],first[np.array(newtris)]]).ravel()
Vp=wP[allT].reshape(-1,3)
fn=np.cross(Vp[1::3]-Vp[0::3],Vp[2::3]-Vp[0::3]); fn/=np.linalg.norm(fn,axis=1,keepdims=True)
N=np.repeat(fn,3,0)
# untouched faces keep their authored normals
moved=np.zeros(len(wP),bool); moved[np.unique(np.array(newtris))]=True
moved|=np.linalg.norm(wP-P[first],axis=1)>1e-7
old=np.arange(keep.sum())
untouched=~moved[T[keep]].any(1)
for c in range(3):
    N[old[untouched]*3+c]=N0[corner[keep][untouched,c]]
p.attributes.POSITION=A(Vp.astype(np.float32),5126,'VEC3',34962,True)
p.attributes.NORMAL=A(N.astype(np.float32),5126,'VEC3',34962)
if UV is not None: p.attributes.TEXCOORD_0=A(UV[src].astype(np.float32),5126,"VEC2",34962)
p.attributes.JOINTS_0=A(J[src].astype(np.uint8),5121,'VEC4',34962)
p.attributes.WEIGHTS_0=A(W[src].astype(np.float32),5126,'VEC4',34962)
p.indices=A(np.arange(len(Vp),dtype=np.uint16),5123,'SCALAR',34963)
print('skin tris',len(T),'->',len(allT))
# the mouth plug shares corners with the cheeks: move them the same way so no seam opens
shift={tuple(np.round(a,5)):b for a,b in zip(P[first][moved],wP[moved])}
for j,q in enumerate(g.meshes[0].primitives):
    if j==k or g.materials[q.material].name!='SkinF': continue
    Q=acc(g,q.attributes.POSITION).astype(np.float64); hit=0
    for i in range(len(Q)):
        t=shift.get(tuple(np.round(Q[i],5)))
        if t is not None: Q[i]=t; hit+=1
    if hit:
        QI=prims[j]['I'].reshape(-1,3); fq=np.cross(Q[QI[:,1]]-Q[QI[:,0]],Q[QI[:,2]]-Q[QI[:,0]])
        fq/=np.linalg.norm(fq,axis=1,keepdims=True); NQ=np.zeros_like(Q)
        for c in range(3): NQ[QI[:,c]]=fq
        q.attributes.POSITION=A(Q.astype(np.float32),5126,'VEC3',34962,True)
        q.attributes.NORMAL=A(NQ.astype(np.float32),5126,'VEC3',34962)
        print('mouth plug corners moved:',hit)
while len(blob)%4: blob.append(0)
g.buffers[0].byteLength=len(blob); g.set_binary_blob(bytes(blob)); g.save(DST)
