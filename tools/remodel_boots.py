"""Replace the boots of human_female.glb with proper low-poly boots.

Each boot is built from horizontal slices so it is one continuous shape: toe
box, instep sloping up to a narrower ankle, calf shaft and a folded cuff, on a
separate dark sole that is thicker under the heel and lifts at the toe. The
shaft is skinned to the shin bone, the foot to the foot bone, blended around
the ankle, so all animations keep working. Run on the output of
flatten_planes.py.

Usage: python tools/remodel_boots.py IN.glb OUT.glb   (needs numpy, pygltflib)
then:  npx @gltf-transform/cli prune OUT.glb OUT.glb
"""
import sys
import numpy as np
from pygltflib import GLTF2, BufferView, Accessor, Primitive, Attributes, Material, PbrMetallicRoughness
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
N=10          # points around each slice
# horizontal slices of the boot upper (left foot, z<0 is the toe):
#   y, front z, back z, half width, centre x offset (outward +)
SLICES=np.array([
    [0.012,-0.198,0.068,0.047,0.000],   # inside the sole
    [0.040,-0.186,0.070,0.049,0.002],   # toe box top edge
    [0.075,-0.128,0.068,0.047,0.002],   # instep
    [0.110,-0.078,0.063,0.044,0.001],
    [0.145,-0.052,0.057,0.041,0.000],   # ankle, the narrowest
    [0.210,-0.054,0.064,0.046,0.000],
    [0.290,-0.058,0.078,0.053,0.000],   # calf
    [0.398,-0.062,0.078,0.057,0.000],   # under the cuff
    [0.402,-0.076,0.096,0.068,0.000],   # folded cuff
    [0.470,-0.078,0.098,0.070,0.000],
])
SOLE_GROW=0.006     # sole sticks out past the upper
SOLE_TOP_TOE,SOLE_TOP_HEEL=0.016,0.032   # sole is thicker under the heel
TOE_SPRING=0.006    # toe tip lifts off the ground

def outline(y,zf,zb,hw,xo):
    """closed slice outline, point 0 at the toe, going round the outside first"""
    zm=(zf+zb)/2; L=(zb-zf)/2
    th=np.linspace(0,2*np.pi,N,endpoint=False)
    z=zm-L*np.cos(th)
    x=hw*np.sin(th)
    # long slices get a foot shape: widest at the ball, slimmer heel
    u=np.clip((zb-z)/(zb-zf),0,1)                      # 0 heel .. 1 toe
    g=np.interp(u,[0,0.3,0.68,0.9,1],[0.84,0.88,1.0,0.92,0.8])
    f=np.clip((2*L-0.14)/0.1,0,1)
    x=x*(1-f+f*g)+xo
    return np.c_[x,np.full(N,y),z]

def upper(side,xc):
    rings=[outline(*s) for s in SLICES]
    tris=[]
    for a,b in zip(rings[:-1],rings[1:]):
        for i in range(N):
            j=(i+1)%N
            tris+=[(a[i],b[i],b[j]),(a[i],b[j],a[j])]
    # thickness at the opening: a short inner wall
    top=rings[-1]; inner=top.copy(); inner[:,[0,2]]*=0.86; inner[:,2]+=0.14*(SLICES[-1,1]+SLICES[-1,2])/2
    inner[:,1]-=0.02
    for i in range(N):
        j=(i+1)%N
        tris+=[(top[i],top[j],inner[j]),(top[i],inner[j],inner[i])]
    return tris

def sole(side,xc):
    base=SLICES[0]
    o=outline(0,base[1]-SOLE_GROW,base[2]+SOLE_GROW,base[3]+SOLE_GROW,base[4])
    u=np.clip((base[2]+SOLE_GROW-o[:,2])/(base[2]-base[1]+2*SOLE_GROW),0,1)
    top=o.copy(); top[:,1]=np.interp(u,[0,0.3,0.36,1],[SOLE_TOP_HEEL,SOLE_TOP_HEEL,SOLE_TOP_TOE,SOLE_TOP_TOE])
    bot=o.copy(); bot[:,1]=np.interp(u,[0,0.85,1],[0,0,TOE_SPRING])
    top[:,1]+=bot[:,1]
    tris=[]
    for i in range(N):
        j=(i+1)%N
        tris+=[(bot[i],top[i],top[j]),(bot[i],top[j],bot[j])]
    for i in range(1,N-1):
        tris+=[(top[0],top[i+1],top[i]),(bot[0],bot[i],bot[i+1])]
    return tris

def place(tris,side,xc):
    V=np.array(tris,dtype=np.float64).reshape(-1,3)
    V[:,0]=xc+side*V[:,0]*-1       # outline x is "outward" for the left foot (-x)
    V=V.reshape(-1,3,3)
    if side>0: V=V[:,[0,2,1]]      # mirroring flips the winding
    return V

old_k=next(i for i,p in enumerate(g.meshes[0].primitives) if g.materials[p.material].name=='BootsF')
old=prims[old_k]
def ss(a,b,x):
    t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def skin(V):
    """shaft rides the shin, foot rides the foot bone, blended around the ankle"""
    J=np.zeros((len(V),4),np.uint8); W=np.zeros((len(V),4),np.float32)
    x,y,z=V[:,0],V[:,1],V[:,2]
    wf=np.maximum(ss(0.16,0.085,y),ss(-0.045,-0.085,z)*ss(0.21,0.13,y))
    for i in range(len(V)):
        sd='Left' if x[i]<0 else 'Right'
        pairs=[(joints.index(sd+'Foot'),wf[i]),(joints.index(sd+'LowerLeg'),1-wf[i])]
        pairs=sorted([p for p in pairs if p[1]>1e-6],key=lambda p:-p[1])
        for c,(jj,ww) in enumerate(pairs): J[i,c]=jj; W[i,c]=ww
    return J,W
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
def build(Vt):
    V=Vt.reshape(-1,3)
    fn=np.cross(Vt[:,1]-Vt[:,0],Vt[:,2]-Vt[:,0]); ln=np.linalg.norm(fn,axis=1)
    Vt=Vt[ln>1e-12]; fn=fn[ln>1e-12]/ln[ln>1e-12,None]; V=Vt.reshape(-1,3)
    J,W=skin(V)
    at=Attributes(POSITION=A(V.astype(np.float32),5126,'VEC3',34962,True),
                  NORMAL=A(np.repeat(fn,3,0).astype(np.float32),5126,'VEC3',34962),
                  JOINTS_0=A(J,5121,'VEC4',34962),WEIGHTS_0=A(W,5126,'VEC4',34962))
    return at,A(np.arange(len(V),dtype=np.uint16),5123,'SCALAR',34963),len(Vt)

up=[];so=[]
for side,xc in ((-1,-0.10),(1,0.10)):
    up.append(place(upper(side,xc),side,xc)); so.append(place(sole(side,xc),side,xc))
at,ia,n1=build(np.vstack(up))
bp=g.meshes[0].primitives[old_k]; bp.attributes=at; bp.indices=ia
g.materials.append(Material(name='SoleF',pbrMetallicRoughness=PbrMetallicRoughness(
    baseColorFactor=[0.035,0.022,0.016,1],metallicFactor=0.0,roughnessFactor=0.9)))
at,ia,n2=build(np.vstack(so))
g.meshes[0].primitives.append(Primitive(attributes=at,indices=ia,material=len(g.materials)-1))
print(f'boots {len(old["I"])//3} tris -> upper {n1} + soles {n2} = {n1+n2}')
while len(blob)%4: blob.append(0)
g.buffers[0].byteLength=len(blob); g.set_binary_blob(bytes(blob)); g.save(DST)
