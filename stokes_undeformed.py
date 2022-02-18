# Stokes in the deformed configurations

import petsc4py, sys
petsc4py.init(sys.argv)

from ufl import transpose
from petsc4py import PETSc
from dolfin import *
from xii import *
import numpy as np
import sympy as sp
import ulfy

print = PETSc.Sys.Print

GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'


def stokes_system(boundaries, mms, dirichlet_tags, scale):
    '''Auxiliary'''
    mu = Constant(mms['parameters']['mu'])
    # Here we need to take geometry into account
    L = Constant(scale)

    mesh = boundaries.mesh()
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    x, y = SpatialCoordinate(mesh)
    phi = as_vector((x, L*y))
    F = grad(phi)

    Grad = lambda arg: dot(grad(arg), inv(F))
    Div = lambda arg: inner(grad(arg), inv(F))
    J = det(F)
    
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)

    n = FacetNormal(mesh)
        
    all_tags = set((1, 2, 3, 4))
    dirichlet_tags = set(dirichlet_tags)    
    neumann_tags = tuple(all_tags - dirichlet_tags)  # Neumann is full stress
    
    f, sigma = mms['forces']['volume'], mms['forces']['stress']
    u0, p0 = mms['solution']['u'], mms['solution']['p']

    bcs = [[DirichletBC(V, u0, bdries, tag) for tag in dirichlet_tags],
           []]

    if not neumann_tags:
        R = FunctionSpace(mesh, 'R', 0)
        W = [V, Q, R]

        u, p, lm = map(TrialFunction, W)
        v, q, dlm = map(TestFunction, W)

        a = block_form(W, 2)
        a[0][0] = inner(2*mu*sym(Grad(u)), sym(Grad(v)))*J*dx
        a[0][1] = -inner(p, Div(v))*J*dx
        a[1][0] = -inner(q, Div(u))*J*dx
        a[1][2] = inner(q, lm)*J*dx
        a[2][1] = inner(dlm, p)*J*dx
        
        # Let's also get the preconditioner
        a_prec = block_form(W, 2)
        a_prec[0][0] = inner(2*mu*sym(Grad(u)), sym(Grad(v)))*J*dx
        # NOTE: There should be J here but without it we get better cond
        a_prec[1][1] = (1/2/mu)*inner(p, q)*dx
        a_prec[2][2] = (2*mu)*inner(lm, dlm)*J*dx

        bcs.append([])
    else:
        raise ValueError
        W = [V, Q]
        
        u, p = map(TrialFunction, W)
        v, q = map(TestFunction, W)
        
        a = block_form(W, 2)
        a[0][0] = inner(2*mu*sym(Grad(u)), sym(Grad(v)))*J*dx
        a[0][1] = -inner(p, Div(v))*J*dx
        a[1][0] = -inner(q, Div(u))*J*dx
        
        # Let's also get the preconditioner
        a_prec = block_form(W, 2)
        a_prec[0][0] = inner(2*mu*sym(Grad(u)), sym(Grad(v)))*J*dx
        a_prec[1][1] = (1/2/mu)*inner(p, q)*J*dx
        
    L = block_form(W, 1)
    L[0] = inner(f, v)*J*dx
    # Neumann
    L[0] += sum(inner(dot(sigma, dot(transpose(inv(F)), n)), v)*J*ds(tag) for tag in neumann_tags)
    
    A, B, b = map(ii_assemble, (a, a_prec, L))

    A, b = apply_bc(A, b, bcs)
    B, _ = apply_bc(B, b=None, bcs=bcs)

    return A, b, W, B

# --------------------------------------------------------------------

if __name__ == '__main__':
    from stokes_deformed import stokes_mms
    import matplotlib.pyplot as plt
    from block.iterative import MinRes
    from scipy.linalg import eigvalsh
    from block.algebraic.petsc import LU
    import gmshnics
    import os
    
    OptDB = PETSc.Options()
    # System size    
    n0 = OptDB.getInt('ncells', 0)
    nrefs = OptDB.getInt('nrefs', 6)
    mu = OptDB.getScalar('mu', 1E0)
    L = OptDB.getScalar('L', 1E0)
    skip = OptDB.getBool('skip', 0)

    dtags = (1, 2, 3, 4)

    iters_history, eigw_history = [], []
    for k in range(n0, n0+nrefs):
        size = 1/2**k
        # We always compute on reference
        mesh, entity_fs = gmshnics.gRectangle((0, 0), (1, 1), size)
        print('Aspect ratio', L)
        bdries = entity_fs[1]

        udef_mms = stokes_mms(mesh, params={'mu': mu, 'L': 1})
        # def_mms = stokes_mms(mesh, params={'mu': mu, 'L': L})
        
        A, b, W, B = stokes_system(bdries, mms=udef_mms, dirichlet_tags=dtags, scale=L)        
        dofs = sum(Wi.dim() for Wi in W)
        if dofs < 10_000 and not skip:
            A_, B_ = map(ii_convert, (A, B))
            eigw = eigvalsh(A_.array(), B_.array())
            lmin, lmax = np.sort(np.abs(eigw))[[0, -1]]
            cond = lmax/lmin
            
            eigw_history.append(eigw)

            fig, ax = plt.subplots()
            for eigw in eigw_history:
                plt.plot(np.linspace(0, dofs, len(eigw)), np.sort(np.abs(eigw)), label=str(eigw),
                         marker='x', linestyle='none')
            plt.show()
        else:
            lmin, lmax, cond = -1, -1, -1

        # Solve it
        for i in range(len(W)):
            B[i][i] = LU(B[i][i])
        
        Ainv = MinRes(A=A, precond=B, tolerance=1E-10)

        x = Ainv*b
        
        wh = ii_Function(W)
        for i, whi in enumerate(wh):
            whi.vector()[:] = x[i]

        File(f'wh_L{L}.pvd') << wh[1]
    
        niters = len(Ainv.residuals)

        def_mms = stokes_mms(mesh, params={'mu': mu, 'L': L})
        # Now for the comparison we want to rescale to deformed mesh
        mesh.coordinates()[:, 1] *= L

        errors = def_mms['get_errors'](wh)
        msg = ' '.join([f'{key} = {val:.3E}' for key, val in errors.items()])
        msg = ' '.join([msg, f'niters = {niters}', f'dofs = {dofs}'])
        msg = ' '.join([msg, f'lmin = {lmin:.3E}', f'lmax = {lmax:.3E}', f'cond = {cond:.2f}'])        
        print(GREEN % msg)

    File('uh_undef.pvd') << wh[0]
    File('ph_undef.pvd') << wh[1]   
