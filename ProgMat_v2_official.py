import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from utilfuncs  import computeFilter, implicit_Ke, implicit_Ae, stiffindex4AK
import matplotlib.pyplot as plt
# modified
class ProgDef():
    def __init__(self,nelx,nely,index,value):
        self.nelx, self.nely = nelx,nely;
        self.nel = self.nelx*self.nely;
        self.ndof = 2*(self.nelx+1)*(self.nely+1);
        self.filterRadius = 3
        H, Hs = computeFilter(self.nelx,self.nely,self.filterRadius)
        self.H,self.Hs = jnp.array(H), jnp.array(Hs)
        self.iK, self.jK, self.iA,self.jA = stiffindex4AK(self.nelx,self.nely)
        self.E = 2;self.nu = 0.3;
    
        ut_obj= jnp.zeros((self.ndof,));
        ut_obj = ut_obj.at[index].set(value);# target deformation freedom, with deformation at non-interested freedom set to zero
        index_u_t = jnp.zeros((self.ndof,));
        index_u_t = index_u_t.at[index].set(1);# identity matrix of size (node*2,1) with 1 in the target deformation freedom;
        self.ut_obj , self.index_u_t = ut_obj, index_u_t
        # return self.ut_obj, self.index_u_t
        

    def objective(self,x):
        # INPUT: THE DISTRIBUTION OF CTE
        # OUTPUT: THE ERROR OF THE PROGRAMA DISPLACEMENT TO THE TEST DISPLACEMENT
        def da_ca_(x):
            Ca  =  (self.E/(1-self.nu**2))* jnp.array([[1,self.nu,0],[self.nu,1,0],[0,0,(1-self.nu)/2]]).reshape(-1,order='F');
            da = Ca.reshape(3,3,order='F') @jnp.array([1,1,0]).reshape(-1,1)
            da_ = jnp.dot(x.reshape(-1,1,order='F'),da.T).T; # linear interpolation
            Ca_ = jnp.dot(jnp.ones((self.nel,1)),Ca.reshape(1,-1,order='F')).T 
            return da_, Ca_
        
        def sK_sA(da_,Ca_):
            sK = implicit_Ke(Ca_)
            sA = implicit_Ae(da_)
            return sK, sA
            
        def K_A(sK,sA):
            K = jnp.zeros((self.ndof,self.ndof))
            A = jnp.zeros((self.ndof,(self.nelx+1)*(self.nely+1)));
            K = K.at[(self.iK,self.jK)].add(sK.flatten('F'))
            A = A.at[(self.iA,self.jA)].add(sA.flatten('F'))  
            
            return K,A
        
        def Kuf(K,A):
            u = jnp.zeros((self.ndof,1)); # deformation field
            theta = jnp.ones(((self.nelx+1)*(self.nely+1),1)); # temperater loading

            fix_left = jnp.arange(0,(self.nely+1)*2,1); # left boundary
            fix_right = jnp.arange(self.ndof-(self.nely+1)*2,self.ndof,1); # right boundary
            dof_fix = jnp.concatenate((fix_left,fix_right),axis=0); # fixed left and right boundary
            dof_fix = fix_left; # fixed only left 

            dof_free = jnp.setdiff1d(jnp.arange(0,self.ndof),dof_fix); # free freedom

            # SOLVE 
            Kr = K[dof_free,:][:,dof_free]; # reduced stiffness matrix
            f_t = A@theta # equivalent nodal force vector
            f_r = f_t[dof_free]; # reduced equivalent nodal force vector
            u_t = jnp.linalg.solve(Kr,f_r); # reduced deformation vector
            u = u.at[dof_free].set(u_t).reshape(-1,order='F'); # deformation vector
            # print(u.shape)
            return u
        
        da_ , Ca_ = da_ca_(x);
        Ke, Ae = sK_sA(da_,Ca_);
        K, A = K_A(Ke,Ae);
        u = Kuf(K,A);

        obj =  jnp.sqrt(jnp.sum(jnp.power(jnp.abs(u*self.index_u_t - self.ut_obj),2)))
        self.u = u;

        return obj,u
        

# %% TARGET SETTING CASE1: UPPER BOUNDARY DEFORMATION
# upper bounary deformation  freedom
nely = 6; nelx = 15;
node_upper = jnp.arange(1,(nely+1)*nelx+1+1,nely+1);
dof_upper = jnp.concatenate((node_upper*2-1,node_upper*2),axis=0)-1;
dof_upper_uy = node_upper *2-1;
dof_upper_ux = node_upper *2-1-1;

value_upper_uy = jnp.linspace(0,0.1,nelx+1)**2;
value_upper_ux = jnp.zeros((len(dof_upper_ux),));

defo_index = jnp.concatenate((dof_upper_ux,dof_upper_uy),axis=0);
defo_value = jnp.concatenate((value_upper_ux,value_upper_uy),axis=0);

nelx = 15; nely = 6;
opti = ProgDef(nelx,nely,defo_index,defo_value);



# %% MMA INITILIZATION
from MMA import mmasub,subsolv
m = 1 
n = nelx* nely; 

x =np.ones(((nelx)*(nely),1))*0.001;  #initial guess
xval = x;
xmin = np.zeros((n,1))# lower bound
xmax = np.ones((n,1))*0.002 # upper bound
xold1 = xval.copy() 
xold2 = xval.copy() 
low = np.ones((n,1))
upp = np.ones((n,1))
a0 = 1.0 
a = np.zeros((m,1)) 
c = 10000*np.ones((m,1))
d = np.zeros((m,1))
move = 0.2 

# %%  OPTIMIZATION
change = 1; loop = 0;
while (change>0.0000000001) and (loop<40):
    loop = loop+1;

    # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    x = jnp.array(x).reshape(-1,1,order='F');
    obj,u =opti.objective(x);
    jacobian_fn = jax.jacobian(opti.objective, argnums=0);
    dobj_dx = jacobian_fn(x)[0];
    
    # FILTERING

    # OPTIMIZATION
    mu0 = 1.0 # Scale factor for objective function
    mu1 = 1.0 # Scale factor for volume constraint function
    f0val = mu0*obj; # [1,1] objective function value
    df0dx = mu0*dobj_dx.reshape(-1,1,order='F') # gradient of the objective function
    df0dx = np.array(df0dx) # jnp to numpy, (n,1)
    fval = np.array([0])[np.newaxis]; # [1,1] constraint function value, not used here
    dfdx =np.zeros([1,n]); # [1,n] gradient of the constraint function, not used here
    xval = np.array(x).reshape(-1,1,order='F') ; # jnp to numpy, (n,1)
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
        mmasub(m,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)
    xold2 = xold1.copy()
    xold1 = xval.copy()
    x = xmma.copy()
    change = np.max(np.abs(x-xold1))
    print('loop = ',loop,'change = ',change,'obj = ',obj)

    # PLOT, clear the plot and plot the new one
    plt.clf()
    plt.plot(value_upper_ux+np.arange(0,nelx+1),value_upper_uy,'o-')
    plt.plot( u[dof_upper_ux].reshape(-1)+np.arange(0,nelx+1),u[dof_upper_uy].reshape(-1),'k-o',linewidth=1,markersize=6)
    # set y axis
    # plt.ylim(0,0.012)
    plt.xlim(0,nelx+1)
    plt.ylabel('Uy')
    plt.xlabel('Ux')
    plt.legend(['target','optimized'])
    plt.pause(0.01)
        


# %% PLOT
# display the deformation field x = value_upper_ux,y = value_upper_uy

import matplotlib.pyplot as plt
import numpy as np
ux = u.at[dof_upper_ux].get().reshape(-1);  
uy = u.at[dof_upper_uy].get().reshape(-1); 
ux =ux+np.arange(0,nelx+1);
plt.figure()
plt.plot(value_upper_ux+np.arange(0,nelx+1),value_upper_uy,'o-')
plt.plot( ux,uy,'k-o',linewidth=1,markersize=6)
# set y axis
# plt.ylim(0,0.012)
plt.xlim(0,nelx+1)
plt.ylabel('Uy')
plt.xlabel('Ux')
plt.legend(['target','optimized'])
plt.show()


plt.figure()
plt.imshow(x.reshape(nelx,nelx), 'jet')
plt.colorbar()
plt.show()


