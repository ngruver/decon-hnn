import numpy as np
import torch
import networkx as nx
import numpy as np
from oil.utils.utils import export,FixedNumpySeed
from src.systems.rigid_body import RigidBody, BodyGraph, project_onto_constraints
from src.animation import Animation
import copy
from src.systems.rigid_body import RigidBody,BodyGraph
from src.systems.chain_pendulum import PendulumAnimation

@export
class SpringPendulum(RigidBody):
    d=3
    dt=.03
    integration_time=3
    g=9.81
    def __init__(self, bobs=2, m=None, l=1,k=10):
        self.body_graph = BodyGraph()#nx.Graph()
        self.arg_string = f"n{bobs}m{m or 'r'}l{l}"
        with FixedNumpySeed(0):
            ms = [.6+.8*np.random.rand() for _ in range(bobs)] if m is None else bobs*[m]
        self.ms = copy.deepcopy(ms)
        ls = bobs*[l]
        self.ks = torch.tensor(bobs*[k]).float()
        #equilibrium positions of the springs
        self.locs = torch.zeros(bobs,3)
        self.locs[:,2] = -torch.arange(bobs).float()
        
        for i in range(bobs):
            self.body_graph.add_extended_nd(i, m=ms.pop(), d=0)
        self.n = bobs
        self.D = self.d*self.n # Spherical coordinates, phi, theta per bob
        self.angular_dims = []

    def sample_initial_conditions(self, bs):
        zn = np.zeros((bs,2,self.n,self.d))
        # equilibrium positions of the springs
        Fs = -np.cumsum(self.ms[::-1])[::-1]*self.g
        zn[:,0,:,2] = torch.from_numpy(np.cumsum(Fs/self.ks.data.numpy())).float()
        zn[:,0,:,2] += .8*np.random.randn(bs,self.n)
        zn[:,0,:] += .4*np.random.randn(bs,self.n,self.d)
        zn[:,1] += .6*np.random.randn(bs,self.n,self.d)
        return torch.from_numpy(zn).float()
    
    def global2bodyCoords(self, global_pos_vel):
        """ input (bs,2,n,3) output (bs,2,3n) """
        return global_pos_vel.reshape(*global_pos_vel.shape[:2],-1)
    def body2globalCoords(self, global_pos_vel):
        """ input (bs,2,3n) output (bs,2,n,3) """
        return global_pos_vel.reshape(*global_pos_vel.shape[:2],-1,3)

    def potential(self, x):
        """inputs [x (bs,n,d)] Gravity potential
           outputs [V (bs,)] """
        gpe = 9.81*(self.M.float() @ x)[..., 2].sum(1)
        l0s = ((self.locs[1:]-self.locs[:-1])**2).sum(-1).sqrt().to(x.device,x.dtype)
        xdist = ((x[:,1:,:]-x[:,:-1,:])**2).sum(-1).sqrt()
        ks = self.ks.to(x.device,x.dtype)
        spring_energy = (.5*ks[1:]*(xdist-l0s)**2).sum(1)
        spring_energy += .5*ks[0]*((x[:,0,:]**2).sum(-1).sqrt()-(self.locs[0]**2).sum().sqrt())**2
        return gpe+spring_energy

    @property
    def animator(self):
        return SpringPendulumAnimation

class PendulumAnimation(Animation):
    def __init__(self, qt,*args,**kwargs):
        super().__init__(qt,*args,**kwargs)
        empty = self.qt.shape[-1] * [[]]
        self.objects["pts"] = sum([self.ax.plot(*empty, "o", ms=10,c=self.colors[i]) for i in range(self.qt.shape[1])], [])

    def update(self, i=0):
        return super().update(i)

def helix(Ns=1000,radius=.05,turns=25):
    t = np.linspace(0,1,Ns)
    xyz = np.zeros((Ns,3))
    xyz[:,0] = np.cos(2*np.pi*Ns*t*turns)*radius
    xyz[:,1] = np.sin(2*np.pi*Ns*t*turns)*radius
    xyz[:,2] = t
    xyz[:,:2][(t>.9)|(t<.1)]=0
    return xyz

def align2ref(refs,vecs):
    """ inputs [refs (n,3), vecs (N,3)]
        outputs [aligned (n,N,3)]
    assumes vecs are pointing along z axis"""
    n,_ = refs.shape
    N,_ = vecs.shape
    norm = np.sqrt((refs**2).sum(-1))
    v = refs/norm[:,None]
    A = np.zeros((n,3,3))
    A[:,:,2] += v
    A[:,2,:] -= v
    M = (np.eye(3)+A+(A@A)/(1+v[:,2,None,None]))
    scaled_vecs = vecs[None]+0*norm[:,None,None] #broadcast to right shape
    scaled_vecs[:,:,2] *= norm[:,None]#[:,None,None]
    return (M[:,None]@scaled_vecs[...,None]).squeeze(-1)

    
class SpringPendulumAnimation(PendulumAnimation):
    
    def __init__(self, *args, spring_lw=.6,spring_r=.2,**kwargs):
        super().__init__(*args, **kwargs)
        empty = self.qt.shape[-1]*[[]]
        self.objects["springs"] = self.ax.plot(*empty,c='k',lw=spring_lw)#
        #self.objects["springs"] = sum([self.ax.plot(*empty,c='k',lw=2) for _ in range(self.n-1)],[])
        self.helix = helix(200,radius=spring_r,turns=10)
        
    def update(self,i=0):
        qt_padded = np.concatenate([0*self.qt[i,:1],self.qt[i,:]],axis=0)
        diffs = qt_padded[1:]-qt_padded[:-1]
        x,y,z = (align2ref(diffs,self.helix)+qt_padded[:-1][:,None]).reshape(-1,3).T
        self.objects['springs'][0].set_data(x,y)
        self.objects['springs'][0].set_3d_properties(z)
        return super().update(i)