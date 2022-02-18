# Stokes in the deformed configurations

import petsc4py, sys
petsc4py.init(sys.argv)

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


def stokes_mms(mesh, params):
    '''Manufactured solution for [] case'''
    # Expect all
    if params is not None:
        mu, L = Constant(params['mu']), Constant(params['L'])
    # All is one
    else:
        params = {'mu': 1.0, 'L': 1.0}
        return stokes_mms(mesh, params)

    x, y = SpatialCoordinate(mesh)
    y = y/L
    
    phi = sin(pi*(x-y))
    u = as_vector((phi.dx(1), -phi.dx(0)))

    p = cos(pi*(x+y))  # NOTE: we wont have zero mean so need to compare proper

    sigma = 2*mu*sym(grad(u)) - p*Identity(len(u))
    f = -div(sigma)

    mu_, L_ = sp.symbols('mu, L')

    subs = {mu: mu_, L: L_}
    
    as_expression = lambda f, subs=subs, params=params: ulfy.Expression(f, subs=subs, degree=4,
                                                                        **params)

    u, p, sigma = map(as_expression, (u, p, sigma))
    
    def get_errors(wh, w=(u, p), norm_ops=None):
        uh, ph = wh[:2]

        u, p = w
        if len(wh) == 3:
            # Compare functions with zero mean
            Qh = ph.function_space()
            dxOmega = dx(domain=Qh.mesh())
            p_mean = assemble(p*dxOmega)/assemble(Constant(1)*dxOmega)

            p = interpolate(p, FunctionSpace(Qh.mesh(), 'DG', 4))
            p.vector()[:] -= p_mean
        
        results = {'|eu|_1': errornorm(u, uh, 'H1', degree_rise=2),
                   '|ep|_0': errornorm(p, ph, 'L2', degree_rise=2)}
        return results

    return {'parameters': params,
            'get_errors': get_errors,
            'forces': {'volume': f, 'stress': sigma},
            'solution': {'u': u, 'p': p}}


def stokes_system(boundaries, mms, dirichlet_tags):
    '''Auxiliary'''
    mu = Constant(mms['parameters']['mu'])

    mesh = boundaries.mesh()
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    
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
        a[0][0] = inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
        a[0][1] = -inner(p, div(v))*dx
        a[1][0] = -inner(q, div(u))*dx
        a[1][2] = inner(q, lm)*dx
        a[2][1] = inner(dlm, p)*dx
        
        # Let's also get the preconditioner
        a_prec = block_form(W, 2)
        a_prec[0][0] = inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
        a_prec[1][1] = (1/2/mu)*inner(p, q)*dx
        a_prec[2][2] = (2*mu)*inner(lm, dlm)*dx

        bcs.append([])
    else:
        W = [V, Q]
        
        u, p = map(TrialFunction, W)
        v, q = map(TestFunction, W)
        
        a = block_form(W, 2)
        a[0][0] = inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
        a[0][1] = -inner(p, div(v))*dx
        a[1][0] = -inner(q, div(u))*dx
        
        # Let's also get the preconditioner
        a_prec = block_form(W, 2)
        a_prec[0][0] = inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
        a_prec[1][1] = (1/2/mu)*inner(p, q)*dx
        
    L = block_form(W, 1)
    L[0] = inner(f, v)*dx
    # Neumann
    L[0] += sum(inner(dot(sigma, n), v)*ds(tag) for tag in neumann_tags)

    A, B, b = map(ii_assemble, (a, a_prec, L))
    print(b[0].norm('l2'))        
    A, b = apply_bc(A, b, bcs)
    B, _ = apply_bc(B, b=None, bcs=bcs)

    return A, b, W, B

# --------------------------------------------------------------------

if __name__ == '__main__':
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

        mesh, entity_fs = gmshnics.gRectangle((0, 0), (1, L), size)
        print('Aspect ratio', L)
        bdries = entity_fs[1]

        mms = stokes_mms(mesh, params={'mu': mu, 'L': L})

        A, b, W, B = stokes_system(bdries, mms=mms, dirichlet_tags=dtags)        
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

        errors = mms['get_errors'](wh)
        msg = ' '.join([f'{key} = {val:.3E}' for key, val in errors.items()])
        msg = ' '.join([msg, f'niters = {niters}', f'dofs = {dofs}'])
        msg = ' '.join([msg, f'lmin = {lmin:.3E}', f'lmax = {lmax:.3E}', f'cond = {cond:.2f}'])        
        print(GREEN % msg)

    File('uh_def.pvd') << wh[0]
    File('ph_def.pvd') << wh[1]    
