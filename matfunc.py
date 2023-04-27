
# import jax.numpy as jnp
import numpy as np
import jax.numpy as jnp
def implicit_Ke(ca):
    # ca is the 9*1 elastic matrix, or a 9*n elastic matrix, which for the all 
    # elements

    # INPUTS:
    # ca: 9*1 or 9*n array, the elastic matrix of candidate material
    # OUTPUTS:
    # Ke: 24*n array, the element stiffness matrix
    t2 = ca [0,: ]/3.0 
    t3 = ca [1,: ]/4.0 
    t4 = ca [2,: ]/3.0 
    t5 = ca [0,: ]/6.0 
    t6 = ca [2,: ]/4.0 
    t7 = ca [3,: ]/4.0 
    t8 = ca [4,: ]/3.0 
    t9 = ca [2,: ]/6.0 
    t10 = ca [5,: ]/3.0 
    t11 = ca [5,: ]/4.0 
    t12 = ca [6,: ]/3.0 
    t13 = ca [4,: ]/6.0 
    t14 = ca [6,: ]/4.0 
    t15 = ca [7,: ]/3.0 
    t16 = ca [5,: ]/6.0 
    t17 = ca [7,: ]/4.0 
    t18 = ca [8,: ]/3.0 
    t19 = ca [6,: ]/6.0 
    t20 = ca [8,: ]/4.0 
    t21 = ca [7,: ]/6.0 
    t22 = ca [8,: ]/6.0 
    t23 = -t2 
    t24 = -t3 
    t25 = -t4 
    t26 = -t5 
    t27 = -t6 
    t28 = -t7 
    t29 = -t8 
    t30 = -t9 
    t31 = -t10 
    t32 = -t11 
    t33 = -t12 
    t34 = -t13 
    t35 = -t14 
    t36 = -t15 
    t37 = -t16 
    t38 = -t17 
    t39 = -t18 
    t40 = -t19 
    t41 = -t20 
    t42 = -t21 
    t43 = -t22 
    t44 = t2+t6+t14+t18 
    t45 = t3+t4+t15+t20 
    t46 = t7+t10+t12+t20 
    t47 = t8+t11+t17+t18 
    t48 = t2+t18+t27+t35 
    t49 = t4+t15+t24+t41 
    t50 = t14+t22+t23+t27 
    t51 = t6+t22+t23+t35 
    t52 = t5+t14+t27+t39 
    t53 = t5+t6+t35+t39 
    t54 = t20+t21+t24+t25 
    t55 = t9+t20+t24+t36 
    t56 = t3+t21+t25+t41 
    t57 = t3+t9+t36+t41 
    t58 = t6+t14+t26+t43 
    t59 = t10+t12+t28+t41 
    t60 = t3+t20+t30+t42 
    t61 = t8+t18+t32+t38 
    t62 = t19+t20+t28+t31 
    t63 = t16+t20+t28+t33 
    t64 = t7+t19+t31+t41 
    t65 = t7+t16+t33+t41 
    t66 = t17+t22+t29+t32 
    t67 = t11+t22+t29+t38 
    t68 = t13+t17+t32+t39 
    t69 = t11+t13+t38+t39 
    t70 = t7+t20+t37+t40 
    t71 = t11+t17+t34+t43 
    t72 = t26+t27+t35+t43 
    t73 = t24+t30+t41+t42 
    t74 = t28+t37+t40+t41 
    t75 = t32+t34+t38+t43 


    Ke = jnp.vstack((t44,t45, t51, t56, t72, t73 ,t52, t55, t46 ,t47 ,\
    t63, t68, t74, t75, t64, t67, t50, t54, t48, t49, t53, t57, t58, t60, t65, t69,\
    t59, t61, t62, t66, t70, t71, t72, t73, t52, t55, t44, t45, t51, t56, t74, t75, t64, t67, t46, t47, t63,\
    t68, t53, t57, t58, t60, t50, t54, t48, t49, t62, t66, t70, t71, t65, t69, t59, t61),dtype=jnp.float32)

    Ke =jnp.ravel(Ke, order='F')

    return Ke

def implicit_Ae(da):
# where the are component of elastic matrix
# the input 'da' is the 3 by (nx^ny) matrix, with each colume inputs
# INPUTS:
# da: 3*nx*ny array, the are component of elastic matrix
# OUTPUTS:
# Ae: 24*n array, the element stiffness matrix
    t2 = da[0,:]/6.0
    t3 = da[1,:]/6.0
    t4 = da[2,:]/6.0
    t8 = da[0,:]/12.
    t9 = da[1,:]/12.
    t10 = da[2,:]/12.
    t5 = -t2
    t6 = -t3
    t7 = -t4
    t11 = -t8
    t12 = -t9
    t13 = -t10
    Ae = jnp.vstack((t5+t7,t6+t7,t2+t13,t4+t12,t8+t10,t9+t10,t4+t11,t3+t13,\
    t5+t13,t7+t12,t2+t7,t4+t6,t4+t8,t3+t10,t10+t11,t9+t13,t11+t13,t12+t13,\
    t7+t8,t6+t10,t2+t4,t3+t4,t5+t10,t7+t9,t7+t11,t6+t13,t8+t13,t10+t12,t2+t10,t4+t9,t4+t5,t3+t7),dtype=jnp.float32)
    Ae = jnp.ravel(Ae, order='F')
    return Ae


def stiffindex4AK(nx,ny):
    # Indexing vectors for 'A' and 'K'
    # INPUTS:
    # nx: int, number of elements in x direction
    # ny: int, number of elements in y direction
    # OUTPUTS:
    # iK: 24*24*n array, the index of element stiffness matrix
    # jK: 24*24*n array, the index of element stiffness matrix
    # iA: 24*24*n array, the index of element stiffness matrix
    # jA: 24*24*n array, the index of element stiffness matrix

    xgrid, ygrid = np.meshgrid(np.arange(1,nx+1), np.arange(1,ny+1))
    n1 = (ny+1)*(xgrid-1) + ygrid
    n1 = n1.reshape(-1,1,order='F')
    n2 = (ny+1)*xgrid + ygrid
    n2 = n2.reshape(-1,1,order='F')
    edof4 = np.c_[n1+1, n2+1, n2, n1]
    edof8 = np.c_[2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, 2*n2-1, 2*n2, 2*n1-1, 2*n1]
    iK =tuple( ((np.kron(edof8, np.ones((8, 1)))).T - 1).flatten('F').astype(int))
    jK = tuple(((np.kron(edof8, np.ones((1, 8)))).T - 1).flatten('F').astype(int))
    iA = tuple(((np.kron(edof8, np.ones((4, 1)))).T - 1).flatten('F').astype(int))
    jA = tuple(((np.kron(edof4, np.ones((1, 8)))).T - 1).flatten('F').astype(int))

    # jK = ((np.kron(edof8, np.ones((1, 8)))).T - 1).flatten('F')
    # iA = ((np.kron(edof8, np.ones((4, 1)))).T - 1).flatten('F')
    # jA = ((np.kron(edof4, np.ones((1, 8)))).T - 1).flatten('F') 
    return iK, jK, iA,jA

def devide_stiff_matrix(nelx,nely):
    # INPUTS:
    # nelx: int, number of elements in x direction
    # nely: int, number of elements in y direction
    # OUTPUTS:
    # allfreedom: 1*2*(nx+1)*(ny+1) array, all freedom
    # alldofs: 1*2*(nx+1)*(ny+1) array, all deformation freedom
    # d1: 8*1 array, nodes opposite lines, and displacement freedom
    # d2: (nely-1+nelx-1)*2*1 array, nodes opposite lines, and displacement freedom
    # d3: 2*1 array, nodes opposite lines, and displacement freedom
    # d4: 2*1 array, nodes opposite lines, and displacement freedom

    alldofs = np.arange(1, 2*(nely+1)*(nelx+1)+1) # all deformation freedom
    nodecorner = np.array([1,nely+1,(nely+1)*nelx+1,(nely+1)*(nelx+1)]) 
    d1 = (np.c_[nodecorner*2-1,nodecorner*2]).reshape(8,1,order='F')
    # nodes opposite lines, and displacement freedom
    node_left_down = np.r_[np.arange(2, nely+1),np.arange((nely+1)*2, (nely+1)*nelx+1, nely+1)]
    d2 = (np.c_[node_left_down*2-1,node_left_down*2]).reshape(((nely-1+nelx-1)*2, 1),order='F')
    node_right_up = np.r_[np.arange((nely+1)*nelx+2, (nely+1)*(nelx+1)),np.arange(nely+2, (nely+1)*(nelx-1)+2, nely+1)]
    d3 = (np.c_[node_right_up*2-1, node_right_up*2]).reshape(((nely-1+nelx-1)*2, 1), order='F')
    # remaining nodes displacement freedom
    d4 = (np.setdiff1d(alldofs, np.concatenate([d1, d2, d3]))).reshape(-1,1,order='F')

    # the index in python start form 0, and are one-d array
    d1 = (d1-1).flatten('F')
    d2 = (d2-1).flatten('F')
    d3 = (d3-1).flatten('F')
    d4 = (d4-1).flatten('F')
    return d1,d2,d3,d4

def test_u(nelx,nely):
    # INPUTS    
    # nelx: number of elements in x direction
    # nely: number of elements in y direction
    # OUTPUTS
    # u: displacement vector
    # f: force vector

    xgrid, ygrid = np.meshgrid(np.arange(0,nelx+1), np.arange(0,nely+1))
    xgrid = xgrid.flatten('F')
    ygrid = ygrid.flatten('F')
    U_testxx = np.zeros(((nelx+1)*(nely+1)*2))
    U_testyy = np.zeros(((nelx+1)*(nely+1)*2))
    U_testxy = np.zeros(((nelx+1)*(nely+1)*2))

    # xx deformation, array
    U_testxx[::2,] = xgrid
    U_testxx[1::2,] = 0

    # yy deformation, array
    U_testyy[::2,] = 0
    U_testyy[1::2,] = - ygrid

    # xy deformation, array
    U_testxy[::2,] = -ygrid[:]
    U_testxy[1::2,] = 0


    # displacement vector
    U_testxx = U_testxx.reshape(-1,1,order='F')
    U_testyy = U_testyy.reshape(-1,1,order='F')
    U_testxy = U_testxy.reshape(-1,1,order='F') 
    ut = np.c_[U_testxx,U_testyy,U_testxy] # displacement vector
   

    
    return ut

def B_matrix():
# displacement-strain matrix, the length of the element is 1, the width is 1. 4 node element.
 
# gauss point is at (1/2,1/2)

    x1 = 1./2.
    y1 = 1./2.
    Bme = np.array([[y1 - 1,     0, 1 - y1,     0, y1, 0,    -y1,     0],
                    [0, x1 - 1,     0,    -x1, 0, x1,     0, 1 - x1],
                    [x1 - 1, y1 - 1,    -x1, 1 - y1, x1, y1, 1 - x1,    -y1]])
    return Bme



# from jax.experimental import sparse

# def spsolve(row, col, data, b):
#   A = sparse.coo.COO((data, row, col), shape=(size,) * 2)
#   return jax.scipy.sparse.linalg.cg(partial(sparse.coo.coo_matvec, A), b)


