"""Reduce the polygon count of human_female.glb while keeping its low-poly look.

Welds each flat-shaded part, simplifies it with meshoptimizer (skin weights as
attributes, open borders locked), then splits it again with flat normals.
Locked areas: the visible face and ears (so the eyes/brows/lips stay seated),
skin under the shoulder strap, and skin touching other layers. The new face
parts from stylize_face.py are already minimal and are left alone.

Usage: python tools/decimate.py IN.glb OUT.glb   (needs numpy, pygltflib, meshoptimizer)
then:  npx @gltf-transform/cli prune OUT.glb OUT.glb
"""
import sys
import numpy as np
import ctypes
import meshoptimizer as mo
from meshoptimizer._loader import lib

def simplify(dst,I,P,attr,aw,lock,target,err,opts):
    fp=lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    up=lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
    lib.meshopt_simplifyWithAttributes.restype=ctypes.c_size_t
    return lib.meshopt_simplifyWithAttributes(up(dst),up(I),ctypes.c_size_t(len(I)),fp(P),ctypes.c_size_t(len(P)),ctypes.c_size_t(12),
        fp(attr),ctypes.c_size_t(attr.shape[1]*4),fp(aw),ctypes.c_size_t(len(aw)),lock.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.c_size_t(target),ctypes.c_float(err),ctypes.c_uint(opts),ctypes.POINTER(ctypes.c_float)())
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
# fraction of triangles to keep per material; parts not listed are left alone
KEEP={'SkinF':0.55,'HairF':0.65,'TopF':0.6,'PantsF':0.5,'BootsF':0.5,'LeatherF':0.6}
ERR=0.015
NJ=len(joints)

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

for k,p in enumerate(g.meshes[0].primitives):
    name=g.materials[p.material].name
    if name not in KEEP: continue
    pr=prims[k]; P=pr['P']; I=pr['I']; J=pr['J']; W=pr['W']
    UV=acc(g,p.attributes.TEXCOORD_0)
    # weld by position (flat-shaded meshes duplicate every corner)
    key=np.round(P,5)
    _,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True)
    inv=inv.ravel()
    wP=P[first].astype(np.float32); wI=inv[I].astype(np.uint32)
    dense=np.zeros((len(first),NJ),np.float32)
    for c in range(4): np.add.at(dense,(np.arange(len(first)),J[first,c]),W[first,c])
    # lock the face so the eyes/brows/lips stay seated on the sculpted surface
    lock=np.zeros(len(first),np.uint8)
    if name=='SkinF':
        x,y,z=wP[:,0],wP[:,1],wP[:,2]
        face=(y>1.54)&(y<1.73)&(z<-0.02)                  # whole visible face incl. jaw
        ears=(y>1.60)&(y<1.71)&(np.abs(x)>0.075)&(z<0.04)
        lock[face|ears]=1
    # lock where another layer sits on top (straps, belt, cuffs) so nothing pokes through
    others=np.vstack([q['P'] for j,q in enumerate(prims) if j!=k and g.materials[g.meshes[0].primitives[j].material].name not in ('SkinF',name)])
    near=np.zeros(len(wP),bool)
    for c in range(0,len(wP),256):
        d=((wP[c:c+256,None,:]-others[None])**2).sum(-1).min(1)
        near[c:c+256]=d<0.012**2
    if name=='SkinF':
        lock[near]=1
        hw=dense[:,joints.index('Head')]
        lock[(hw<0.5)&(wP[:,1]>1.30)&(np.abs(wP[:,0])<0.21)]=1   # shoulders/chest under the strap
    dst=np.zeros_like(wI)
    tgt=int(len(wI)*KEEP[name])//3*3
    n=simplify(dst,wI,np.ascontiguousarray(wP),np.ascontiguousarray(dense),np.full(NJ,0.15,np.float32),
        lock,tgt,ERR,mo.SIMPLIFY_LOCK_BORDER)
    T=dst[:n].reshape(-1,3)
    # unweld again with flat normals to keep the faceted look
    V=wP[T].reshape(-1,3)
    fn=np.cross(V[1::3]-V[0::3],V[2::3]-V[0::3]); fn/=np.linalg.norm(fn,axis=1,keepdims=True)+1e-12
    N=np.repeat(fn,3,0).astype(np.float32)
    src=first[T.ravel()]
    # triangles that survived untouched keep their original corners (normals included)
    N0=acc(g,p.attributes.NORMAL); OT=I.reshape(-1,3)
    orig={tuple(sorted(inv[t])):t for t in OT}
    for ti,t in enumerate(T):
        o=orig.get(tuple(sorted(t)))
        if o is None: continue
        o=[c for w_ in t for c in o if inv[c]==w_]   # align corner order with t
        src[ti*3:ti*3+3]=o; N[ti*3:ti*3+3]=N0[o]
    p.attributes.POSITION=A(V.astype(np.float32),5126,'VEC3',34962,True)
    p.attributes.NORMAL=A(N,5126,'VEC3',34962)
    p.attributes.TEXCOORD_0=A(UV[src].astype(np.float32),5126,'VEC2',34962)
    p.attributes.JOINTS_0=A(J[src].astype(np.uint8),5121,'VEC4',34962)
    p.attributes.WEIGHTS_0=A(W[src].astype(np.float32),5126,'VEC4',34962)
    p.indices=A(np.arange(len(V),dtype=np.uint16),5123,'SCALAR',34963)
    print(f'{name:9s} {len(I)//3:5d} -> {n//3:5d} tris')

while len(blob)%4: blob.append(0)
g.buffers[0].byteLength=len(blob); g.set_binary_blob(bytes(blob)); g.save(DST)
