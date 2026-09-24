"""Restyle the face of human_female.glb: black dot eyes with a glint, blocky
brows, faceted geometric lips and stronger brow/cheek/nose planes.

Run on the output of refine_human_female.py; all new parts are skinned to the
Head bone so every animation keeps working. The old anime eye meshes are
dropped and the textured mouth becomes a skin-coloured plug behind the lips.

Usage: python tools/stylize_face.py IN.glb OUT.glb   (needs numpy, pygltflib)
then:  npx @gltf-transform/cli prune OUT.glb OUT.glb  (drops unused data)
"""
import sys
import numpy as np
import sys
import numpy as np
from pygltflib import GLTF2, Material, PbrMetallicRoughness, Primitive, Attributes, Accessor, BufferView
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
def raycast_z(P,I,pts,dirz=1.0,z0=-1.0):
    """First hit along +z from (x,y,z0). Returns z or nan."""
    T=P[I.reshape(-1,3)].astype(np.float64)
    v0,v1,v2=T[:,0],T[:,1],T[:,2]; e1=v1-v0; e2=v2-v0
    d=np.array([0,0,dirz])
    out=[]
    for x,y in pts:
        o=np.array([x,y,z0])
        h=np.cross(d,e2); a=(e1*h).sum(1); ok=np.abs(a)>1e-12
        f=np.where(ok,1/np.where(ok,a,1),0); s=o-v0
        u=f*(s*h).sum(1); qv=np.cross(s,e1); v=f*(qv@d); t=f*(e2*qv).sum(1)
        hit=ok&(u>=0)&(v>=0)&(u+v<=1)&(t>0)
        out.append(z0+dirz*t[hit].min() if hit.any() else np.nan)
    return np.array(out)
def ss(a,b,x):
    t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def g2(x,y,cx,cy,sx,sy): return np.exp(-((x-cx)/sx)**2-((y-cy)/sy)**2)

MAT={m.name:i for i,m in enumerate(g.materials)}
PRIM={g.materials[p.material].name:i for i,p in enumerate(g.meshes[0].primitives)}
HEAD=joints.index('Head')

# ------------------------------------------------------------------ sculpt
def sculpt(P,w):
    x,y,z=P[:,0].astype(np.float64),P[:,1].astype(np.float64),P[:,2].astype(np.float64)
    ax=np.abs(x); front=ss(-0.06,-0.085,z)*w
    D=np.zeros((len(P),3))
    # brow ridge shelf over the eyes
    D[:,2]-=front*0.006*g2(ax,y,0.034,1.693,0.028,0.007)
    # shallow eye sockets so the eye domes sit in the face
    D[:,2]+=front*0.004*g2(ax,y,0.036,1.667,0.016,0.009)
    # cheekbones: broad forward/outward plane
    cb=front*g2(ax,y,0.052,1.638,0.016,0.012)
    D[:,2]-=cb*0.005; D[:,0]+=np.sign(x)*cb*0.002
    # nose: broader wedge that widens toward the base, stronger tip
    nz=front*ss(-0.094,-0.100,z)*(1-ss(0.018,0.03,ax))*ss(1.608,1.618,y)*(1-ss(1.662,1.675,y))
    widen=0.10+0.35*(1-ss(1.615,1.66,y))
    D[:,0]+=nz*x*widen
    D[:,2]-=nz*0.004*g2(ax,y,0,1.628,0.012,0.01)
    # chin block
    D[:,2]-=front*0.003*g2(ax,y,0,1.566,0.015,0.008)
    return (P+D).astype(np.float32)

blob=bytearray(g.binary_blob())
def write_acc(i,arr):
    a=g.accessors[i];bv=g.bufferViews[a.bufferView]
    off=(bv.byteOffset or 0)+(a.byteOffset or 0)
    b=arr.astype(np.float32).tobytes(); blob[off:off+len(b)]=b
def face_normals(P,I):
    I=I.reshape(-1,3); fn=np.cross(P[I[:,1]]-P[I[:,0]],P[I[:,2]]-P[I[:,0]])
    vn=np.zeros_like(P)
    for c in range(3): np.add.at(vn,I[:,c],fn)
    return vn/(np.linalg.norm(vn,axis=1,keepdims=True)+1e-12)

for name in ('SkinF','MouthF'):
    k=PRIM[name]; p=g.meshes[0].primitives[k]; pr=prims[k]
    newP=sculpt(pr['P'],boneW(pr,'Head'))
    N=acc(g,p.attributes.NORMAL)
    n0=face_normals(pr['P'].astype(np.float64),pr['I']); n1=face_normals(newP.astype(np.float64),pr['I'])
    ok=(n0*N).sum(1)>0.98
    write_acc(p.attributes.POSITION,newP); write_acc(p.attributes.NORMAL,np.where(ok[:,None],n1,N))
    a=g.accessors[p.attributes.POSITION]; a.min=newP.min(0).tolist(); a.max=newP.max(0).tolist()
    pr['P']=newP

# the mouth quad becomes a skin-coloured plug behind the new geometric lips
g.meshes[0].primitives[PRIM['MouthF']].material=MAT['SkinF']

# surface for projecting new features (skin + mouth plug)
sk,mo=prims[PRIM['SkinF']],prims[PRIM['MouthF']]
SP=np.vstack([sk['P'],mo['P']]); SI=np.concatenate([sk['I'],mo['I']+len(sk['P'])])
def surf(xy): return raycast_z(SP,SI,xy)

# ------------------------------------------------------------------ geometry builders
def flat(tris):
    """list of (a,b,c) -> unindexed positions with flat normals"""
    V=np.array(tris,dtype=np.float64).reshape(-1,3,3)
    n=np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]); ln=np.linalg.norm(n,axis=1)
    V,n=V[ln>1e-10],n[ln>1e-10]; n/=np.linalg.norm(n,axis=1,keepdims=True)
    # make every face point away from the head (towards -z / outward)
    flip=(n[:,2]>0.35)
    V[flip]=V[flip][:,[0,2,1]]; n[flip]*=-1
    return V.reshape(-1,3),np.repeat(n,3,axis=0)

def strip(top,bot,depth,walls=(True,True)):
    """front faces between two polylines + side walls sinking into the face"""
    tris=[]
    for i in range(len(top)-1):
        a,b,c,d=top[i],top[i+1],bot[i+1],bot[i]
        tris+= [(a,b,c),(a,c,d)]
    back=lambda v: v+np.array([0,0,depth])
    if not any(walls): return tris
    for line,on in zip((top,bot),walls):
        if not on: continue
        for i in range(len(line)-1):
            a,b=line[i],line[i+1]; tris+=[(a,back(a),back(b)),(a,back(b),b)]
    for a,d in ((top[0],bot[0]),(top[-1],bot[-1])):
        tris+=[(a,d,back(d)),(a,back(d),back(a))]
    return tris

def brow(side):
    # inner -> arch -> tapered tail (x,y,thickness)
    ctrl=np.array([[0.012,1.6875,0.0060],[0.022,1.6925,0.0062],[0.034,1.6965,0.0056],
                   [0.044,1.6965,0.0046],[0.052,1.6935,0.0032],[0.059,1.6885,0.0012]])
    t=np.linspace(0,1,5); k=np.linspace(0,1,len(ctrl))
    c=np.stack([np.interp(t,k,ctrl[:,j]) for j in range(3)],1)
    tan=np.gradient(c[:,:2],axis=0); tan/=np.linalg.norm(tan,axis=1,keepdims=True)
    nrm=np.stack([-tan[:,1],tan[:,0]],1); nrm*=np.sign(nrm[:,1:2])
    top=c[:,:2]+nrm*c[:,2:3]/2; bot=c[:,:2]-nrm*c[:,2:3]/2
    top[:,0]*=side; bot[:,0]*=side
    lift=0.0028
    T=np.c_[top,surf(top)-lift]; B=np.c_[bot,surf(bot)-lift*0.7]
    return strip(list(T),list(B),0.004)

def eye(side):
    cx,cy=0.036*side,1.667
    zs=surf([(cx,cy)])[0]
    rx,ry,rz=0.0098,0.0120,0.0060
    c=np.array([cx,cy,zs+0.0008])
    nu,nv=8,2
    V=[];N=[]
    for j in range(nv+1):
        th=(np.pi/2)*j/nv            # 0 = front pole
        for i in range(nu):
            ph=2*np.pi*i/nu
            d=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),-np.cos(th)])
            V.append(c+d*[rx,ry,rz]); n=d/np.array([rx,ry,rz]); N.append(n/np.linalg.norm(n))
    idx=[]
    for j in range(nv):
        for i in range(nu):
            a=j*nu+i;b=j*nu+(i+1)%nu;cc=(j+1)*nu+(i+1)%nu;d=(j+1)*nu+i
            idx+=[a,d,cc,a,cc,b] if j>0 else [a,d,cc]
    V=np.array(V);N=np.array(N)
    T=np.array(idx).reshape(-1,3)
    fn=np.cross(V[T[:,1]]-V[T[:,0]],V[T[:,2]]-V[T[:,0]])
    bad=(fn*N[T].sum(1)).sum(1)<0
    T[bad]=T[bad][:,[0,2,1]]; idx=T.ravel()
    # highlight: small disc on the upper dome, same screen side for both eyes (key light)
    hx,hy=-0.0042,0.0048
    u,v=hx/rx,hy/ry; hz=c[2]-rz*np.sqrt(max(0,1-u*u-v*v))-0.0004
    hc=np.array([cx+hx,cy+hy,hz]); r=0.0022
    ring=[hc+[r*np.cos(a),r*1.15*np.sin(a),0.0003*np.cos(a)] for a in np.linspace(0,2*np.pi,5)[:-1]]
    shine=[(ring[0],ring[1],ring[2]),(ring[0],ring[2],ring[3])]
    # small lash flick at the outer-upper corner (feminine cue, same block language as the brows)
    base=[]
    for a in np.linspace(np.radians(40),np.radians(10),2):
        base.append(np.array([cx+side*rx*1.05*np.cos(a),cy+ry*1.05*np.sin(a)]))
    tip=np.array([cx+side*(rx+0.0055),cy+ry*0.55])
    inner=[b+(np.array([cx,cy])-b)*0.18 for b in base]
    outer=base
    top=[np.array([*b,0]) for b in outer]+[np.array([*tip,0])]
    bot=[np.array([*b,0]) for b in inner]+[np.array([*tip,0])]
    top=np.array(top);bot=np.array(bot)
    top[:,2]=surf(top[:,:2])-0.0022; bot[:,2]=surf(bot[:,:2])-0.0022
    lash=strip(list(top),list(bot),0.003,(True,False))
    return (V,N,np.array(idx)),shine,lash

def lips():
    xs=np.array([-0.019,-0.011,-0.004,0,0.004,0.011,0.019]); ax=np.abs(xs)
    line=1.5932-0.0008*(1-(ax/0.019)**2)                 # mouth line, slight downward centre
    bow=np.where(ax<0.004,1.5992-0.0012*(1-ax/0.004),0)  # cupid's bow dip
    up=np.where(ax<0.004,bow,1.6000-0.0068*(np.clip(ax-0.004,0,None)/0.015)**1.6)
    lo=1.5856+0.0072*(ax/0.019)**1.8
    def row(yv,proj):
        xy=np.c_[xs,yv]; return np.c_[xy,surf(xy)-proj]
    prot_u=0.0030*(1-(ax/0.019)**2)+0.0003
    prot_l=0.0042*(1-(ax/0.019)**2)+0.0003
    U0=row(up,0.0006+0*ax); U1=row((up+line)/2,prot_u); U2=row(line+0.0002,prot_u*0.55)
    L0=row(line-0.0002,prot_l*0.55); L1=row((lo+line)/2+0.0006,prot_l); L2=row(lo,0.0006+0*ax)
    upper=strip(list(U0),list(U1),0.003,(True,False))+strip(list(U1),list(U2),0.003,(False,False))
    lower=strip(list(L0),list(L1),0.003,(False,False))+strip(list(L1),list(L2),0.003,(False,True))
    # dark mouth line tucked between the lips
    M0=row(line+0.0005,prot_u*0.3); M1=row(line-0.0005,prot_l*0.3)
    mline=strip(list(M0[::2]),list(M1[::2]),0.002,(False,False))
    return upper+lower,mline

# ------------------------------------------------------------------ assemble
def add_mat(name,rgb,rough=0.8,emis=None):
    m=Material(name=name,pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[*rgb,1.0],metallicFactor=0.0,roughnessFactor=rough))
    if emis: m.emissiveFactor=emis
    g.materials.append(m); return len(g.materials)-1

def push(arr,target=None):
    b=arr.tobytes()
    while len(blob)%4: blob.append(0)
    off=len(blob); blob.extend(b)
    g.bufferViews.append(BufferView(buffer=0,byteOffset=off,byteLength=len(b),target=target))
    return len(g.bufferViews)-1

def add_prim(P,N,mat,idx=None):
    P=np.asarray(P,np.float32);N=np.asarray(N,np.float32);n=len(P)
    def A(arr,ctype,typ,target,mm=False):
        bv=push(arr,target)
        a=Accessor(bufferView=bv,componentType=ctype,count=len(arr),type=typ)
        if mm: a.min=arr.min(0).tolist(); a.max=arr.max(0).tolist()
        g.accessors.append(a); return len(g.accessors)-1
    J=np.zeros((n,4),np.uint8); J[:,0]=HEAD
    W=np.zeros((n,4),np.float32); W[:,0]=1
    UV=np.zeros((n,2),np.float32)
    at=Attributes(POSITION=A(P,5126,'VEC3',34962,True),NORMAL=A(N,5126,'VEC3',34962),
                  TEXCOORD_0=A(UV,5126,'VEC2',34962),JOINTS_0=A(J,5121,'VEC4',34962),WEIGHTS_0=A(W,5126,'VEC4',34962))
    if idx is None: idx=np.arange(n)
    ia=A(np.asarray(idx,np.uint16),5123,'SCALAR',34963)
    g.meshes[0].primitives.append(Primitive(attributes=at,indices=ia,material=mat))

hair=g.materials[MAT['HairF']].pbrMetallicRoughness.baseColorFactor
m_eye=add_mat('EyeDot',[0.012,0.010,0.012],rough=0.22)
m_shine=add_mat('EyeGlint',[1,1,1],rough=0.5,emis=[1,1,1])
m_brow=add_mat('Brow',[hair[0]*0.55,hair[1]*0.55,hair[2]*0.55],rough=0.9)
m_lip=add_mat('Lips',[0.50,0.19,0.16],rough=0.6)
m_line=add_mat('MouthLine',[0.06,0.02,0.02],rough=0.9)

eyeV=[];eyeN=[];eyeI=[];shine=[];lash=[];brows=[]
for side in (1,-1):
    (V,N,I),s,l=eye(side); eyeI.append(I+len(np.vstack(eyeV)) if eyeV else I); eyeV.append(V); eyeN.append(N)
    shine+=s; lash+=l; brows+=brow(side)
add_prim(np.vstack(eyeV),np.vstack(eyeN),m_eye,np.concatenate(eyeI))
add_prim(*flat(shine),m_shine)
add_prim(*flat(lash+brows),m_brow)
lp,ml=lips()
add_prim(*flat(lp),m_lip)
add_prim(*flat(ml),m_line)

# drop the old anime eye parts
drop={PRIM[n] for n in ('EyeF','Shine','LashF','WhiteF')}
g.meshes[0].primitives=[p for i,p in enumerate(g.meshes[0].primitives) if i not in drop]

while len(blob)%4: blob.append(0)
g.buffers[0].byteLength=len(blob)
g.set_binary_blob(bytes(blob))
g.save(DST)
print('ok',DST)
