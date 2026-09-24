"""Reshape the low-poly human_female.glb toward more natural anatomy.

Only vertex positions/normals are changed: topology, UVs, materials, skinning
and animations are untouched, so the model stays drop-in compatible.

Usage: python tools/refine_human_female.py IN.glb OUT.glb   (needs numpy, pygltflib)
"""
import sys
import numpy as np
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
g=GLTF2().load(sys.argv[1])
joints=[g.nodes[j].name for j in g.skins[0].joints]
prims=[]
for p in g.meshes[0].primitives:
    P=acc(g,p.attributes.POSITION);J=acc(g,p.attributes.JOINTS_0);W=acc(g,p.attributes.WEIGHTS_0)
    prims.append(dict(mat=g.materials[p.material].name,P=P,J=J,W=W,I=acc(g,p.indices).ravel()))
def boneW(pr,name):
    b=joints.index(name);return ((pr['J']==b)*pr['W']).sum(1)

par={}
for i,n in enumerate(g.nodes):
    for c in (n.children or []): par[c]=i
def wpos(i):
    p=np.zeros(3)
    while i is not None:
        p=p+np.array(g.nodes[i].translation or [0,0,0]); i=par.get(i)
    return p
JP={g.nodes[j].name:wpos(j) for j in g.skins[0].joints}

def ss(a,b,x):
    t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def band(x,a,b,c,d):  # 0 below a, ramps to 1 at b, 1 until c, ramps to 0 at d
    return ss(a,b,x)*(1-ss(c,d,x))
def gauss(x,c,w): return np.exp(-((x-c)/w)**2)

def limb(P,w,A,B,prof,side):
    """Radial reshaping around bone segment A->B.
    prof: dict of (knots, values) for 'out','in','front','back' scale and 'dz' offset (+ = back)."""
    ax=B-A; L=np.linalg.norm(ax); ax/=L
    t=((P-A)@ax)/L
    C=A+np.outer(t*L,ax); r=P-C
    f=np.array([0,0,-1.0]); f-=ax*(f@ax); f/=np.linalg.norm(f)
    l=np.cross(ax,f); l*=np.sign(l[0]*side) if abs(l[0])>1e-6 else 1
    rl=r@l; rf=r@f; rest=r-np.outer(rl,l)-np.outer(rf,f)
    tt=np.clip(t,0,1)
    def pv(k): kn,v=prof.get(k,([0,1],[1,1])); return np.interp(tt,kn,v)
    rl2=np.where(rl>0,rl*pv('out'),rl*pv('in'))
    rf2=np.where(rf>0,rf*pv('front'),rf*pv('back'))
    dzb=np.interp(tt,*prof['dz']) if 'dz' in prof else 0*tt
    dyv=np.interp(tt,*prof['dy']) if 'dy' in prof else 0*tt
    newP=C+np.outer(rl2,l)+np.outer(rf2,f)+rest
    newP[:,2]+=dzb; newP[:,1]+=dyv
    return (newP-P)*w[:,None]

def deform(pr):
    P=pr['P'].astype(np.float64); W=lambda n: boneW(pr,n)
    x,y,z=P[:,0],P[:,1],P[:,2]; ax_=np.abs(x)
    D=np.zeros_like(P)
    armW={s:W(s+'Shoulder')+W(s+'UpperArm')+W(s+'LowerArm')+W(s+'Hand') for s in ('Left','Right')}
    legW={s:W(s+'UpperLeg')+W(s+'LowerLeg')+W(s+'Foot') for s in ('Left','Right')}
    headW=W('Head'); arms=armW['Left']+armW['Right']; legs=legW['Left']+legW['Right']
    core=np.clip(1-arms-legs-headW,0,1)

    # ---- Neck: thicker, with sternocleidomastoid taper; shorter via trapezius rise
    nk=np.clip(1-headW-arms,0,1)*band(y,1.40,1.46,1.555,1.585)*(np.hypot(x,z)<0.075)
    s=1+0.24*band(y,1.40,1.46,1.50,1.575)
    D[:,0]+=nk*x*(s-1); D[:,2]+=nk*(z-0.004)*(s-1)
    # trapezius: raise upper back/shoulder line near the neck, sloping to the acromion
    trap=np.clip(1-headW,0,1)*ss(1.37,1.445,y)*(1-ss(1.50,1.56,y))
    trap*=(1-ss(0.05,0.23,ax_))*ss(0.0,0.04,ax_+0.04)
    zfac=0.45+0.55*ss(-0.06,0.03,z)          # back rises more than front (clavicle)
    D[:,1]+=0.04*trap*zfac*(1-0.8*arms)
    # neck base flare into the traps (widen lowest neck ring)
    fl=np.clip(1-headW-arms,0,1)*band(y,1.43,1.455,1.47,1.50)*(np.hypot(x,z)<0.08)
    D[:,0]+=fl*x*0.12; D[:,2]+=fl*z*0.08

    # round the boxy top of the shoulder girdle front-to-back
    rt=np.clip(1-headW,0,1)*ss(1.395,1.455,y)*(1-ss(1.47,1.50,y))*(1-ss(0.17,0.22,ax_))*(1-0.7*arms)
    D[:,2]-=rt*(z-0.005)*0.16
    # ---- Shoulders: round the flat shelf corner into a deltoid cap
    for sgn,sd in ((-1,'Left'),(1,'Right')):
        cap=gauss(y,1.415,0.03)*ss(0.13,0.19,sgn*x)*(1-ss(0.23,0.27,sgn*x))*(1-headW)
        D[:,0]+=sgn*0.014*cap
        D[:,1]-=0.008*cap*ss(0.17,0.21,sgn*x)

    # ---- Torso: smoother ribcage-to-waist taper, softer glutes, bust
    tor=core
    sx=np.interp(y,[0.99,1.03,1.09,1.15,1.18,1.21,1.24,1.30,1.36],
                   [1.00,1.03,1.02,1.03,1.06,0.975,0.985,1.00,1.00])
    D[:,0]+=tor*x*(sx-1)
    # glutes: reduce the shelf-like projection and lift the apex slightly
    gl=tor*ss(0.0,0.07,z)
    D[:,2]-=gl*0.020*gauss(y,0.905,0.045)*(1-ss(0.10,0.20,ax_))
    D[:,2]-=gl*0.008*gauss(y,0.85,0.02)*(1-ss(0.10,0.20,ax_))
    # lower-back curve (lordosis) a touch deeper
    D[:,2]-=tor*0.008*gauss(y,1.08,0.05)*ss(0.03,0.07,z)
    # bust: rounder lower pole, gentle upper slope, subtle cleavage
    bu=tor*ss(-0.08,-0.11,z)
    D[:,2]+=bu*(0.010*gauss(y,1.30,0.02)-0.006*gauss(y,1.33,0.02)+0.004*gauss(y,1.24,0.015))
    D[:,2]+=bu*0.012*gauss(x,0,0.028)*band(y,1.22,1.25,1.31,1.345)
    # abdomen: slight belly curve under the navel
    D[:,2]-=tor*0.006*gauss(y,1.02,0.035)*ss(-0.05,-0.08,z)

    # ---- Arms
    for sgn,sd in ((-1,'Left'),(1,'Right')):
        ua=W(sd+'UpperArm')+0.5*W(sd+'Shoulder')
        D+=limb(P,ua,JP[sd+'UpperArm'],JP[sd+'LowerArm'],{
            'out':  ([0,0.12,0.3,0.55,0.8,1],[1.25,1.42,1.28,1.12,1.06,1.0]),   # deltoid
            'in':   ([0,0.12,0.3,0.55,0.8,1],[1.0,1.05,1.12,1.14,1.06,1.0]),
            'front':([0,0.15,0.4,0.6,0.85,1],[1.2,1.3,1.22,1.22,1.08,1.0]),    # biceps
            'back': ([0,0.15,0.4,0.6,0.85,1],[1.2,1.3,1.28,1.2,1.05,1.0]),     # triceps
        },sgn)
        la=W(sd+'LowerArm')
        D+=limb(P,la,JP[sd+'LowerArm'],JP[sd+'Hand'],{
            'out':  ([0,0.2,0.45,0.8,1],[1.0,1.22,1.12,0.95,0.92]),          # brachioradialis
            'in':   ([0,0.2,0.45,0.8,1],[1.0,1.16,1.08,0.94,0.92]),
            'front':([0,0.25,0.5,0.85,1],[1.0,1.14,1.06,0.9,0.88]),
            'back': ([0,0.25,0.5,0.85,1],[1.0,1.12,1.04,0.9,0.88]),
        },sgn)

    # ---- Legs
    for sgn,sd in ((-1,'Left'),(1,'Right')):
        ul=W(sd+'UpperLeg')
        D+=limb(P,ul,JP[sd+'UpperLeg'],JP[sd+'LowerLeg'],{
            'out':  ([0,0.3,0.55,0.8,0.92,1],[1.0,1.02,1.04,0.97,0.93,0.95]),
            'in':   ([0,0.3,0.55,0.75,0.92,1],[1.0,1.0,1.03,1.04,0.92,0.95]),
            'front':([0,0.3,0.6,0.85,0.95,1],[1.0,1.05,1.07,1.0,1.0,1.02]),   # quads
            'back': ([0,0.3,0.6,0.85,0.95,1],[1.0,1.0,1.02,0.95,0.92,0.95]),  # knee pit
        },sgn)
        ll=W(sd+'LowerLeg')
        D+=limb(P,ll,JP[sd+'LowerLeg'],JP[sd+'Foot'],{
            'out':  ([0,0.12,0.3,0.5,0.75,0.9,1],[0.95,1.0,1.08,1.02,0.9,0.88,0.95]),
            'in':   ([0,0.12,0.3,0.5,0.75,0.9,1],[0.95,1.0,1.1,1.02,0.9,0.88,0.95]),
            'front':([0,0.12,0.5,0.9,1],[1.0,1.0,0.96,0.92,1.0]),
            'back': ([0,0.12,0.3,0.45,0.7,0.9,1],[0.95,1.05,1.32,1.2,0.95,0.9,1.0]),  # calf
        },sgn)
        ft=W(sd+'Foot')
        # longer, lower foot with a tapered toe box
        toe=ft*ss(-0.05,-0.13,z)
        D[:,2]-=toe*0.03*ss(-0.05,-0.15,z)/1.0
        D[:,1]-=toe*0.012*ss(0.03,0.08,y)
        D[:,0]+=toe*(x-sgn*0.10)*(-0.06)
        # heel: slight rounding back
        hl=ft*ss(0.02,0.06,z)*(1-ss(0.06,0.12,y))
        D[:,2]+=hl*0.008
    return (P+D).astype(np.float32)

def face_normals(P,I):
    I=I.reshape(-1,3); fn=np.cross(P[I[:,1]]-P[I[:,0]],P[I[:,2]]-P[I[:,0]])
    vn=np.zeros_like(P)
    for c in range(3): np.add.at(vn,I[:,c],fn)
    return vn/(np.linalg.norm(vn,axis=1,keepdims=True)+1e-12)

blob=bytearray(g.binary_blob())
def write_acc(i,arr):
    a=g.accessors[i];bv=g.bufferViews[a.bufferView]
    assert (bv.byteStride or 12)==12
    off=(bv.byteOffset or 0)+(a.byteOffset or 0)
    b=arr.astype(np.float32).tobytes(); blob[off:off+len(b)]=b

for k,p in enumerate(g.meshes[0].primitives):
    pr=prims[k]; newP=deform(pr)
    moved=np.linalg.norm(newP-pr['P'],axis=1)>1e-7
    if not moved.any(): continue
    N=acc(g,p.attributes.NORMAL)
    n0=face_normals(pr['P'].astype(np.float64),pr['I']); n1=face_normals(newP.astype(np.float64),pr['I'])
    ok=(n0*N).sum(1)>0.98       # only replace normals where they were topology-derived
    newN=np.where(ok[:,None],n1,N)
    write_acc(p.attributes.POSITION,newP); write_acc(p.attributes.NORMAL,newN)
    a=g.accessors[p.attributes.POSITION]; a.min=newP.min(0).tolist(); a.max=newP.max(0).tolist()
    print(pr['mat'],'moved',moved.sum(),'max disp',np.linalg.norm(newP-pr['P'],axis=1).max().round(4))
g.set_binary_blob(bytes(blob))
g.save(sys.argv[2])
