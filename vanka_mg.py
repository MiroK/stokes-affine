# Stokes in the deformed configurations

import petsc4py, sys
petsc4py.init(sys.argv)

from firedrake import *
import numpy as np


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

    p = cos(pi*(x+y))  

    sigma = 2*mu*sym(grad(u)) - p*Identity(len(u))
    f = -div(sigma)

    def get_errors(wh, w=(u, p), mean=False):
        uh, ph = wh.split()

        u, p = w

        if mean:
            # Compare functions with zero mean
            Qh = ph.function_space()
            dxOmega = dx(domain=Qh.mesh())
            p_mean = assemble(p*dxOmega)/assemble(Constant(1)*dxOmega)
            
            p = interpolate(p, FunctionSpace(Qh.mesh(), 'DG', 4))
            p -= p_mean
        
        results = {'|eu|_1': errornorm(u, uh, 'H1', degree_rise=2),
                   '|ep|_0': errornorm(p, ph, 'L2', degree_rise=2)}
        return results

    return {'parameters': params,
            'get_errors': get_errors,
            'forces': {'volume': f, 'stress': sigma},
            'solution': {'u': u, 'p': p}}


def stokes_system(mesh, mms, dirichlet_tags):
    '''Auxiliary'''
    mu = Constant(mms['parameters']['mu'])

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = V*Q
    
    n = FacetNormal(mesh)
        
    all_tags = set((1, 2, 3, 4))
    dirichlet_tags = set(dirichlet_tags)    
    neumann_tags = tuple(all_tags - dirichlet_tags)  # Neumann is full stress
    
    f, sigma = mms['forces']['volume'], mms['forces']['stress']
    u0, p0 = mms['solution']['u'], mms['solution']['p']

    bcs = DirichletBC(W.sub(0), u0, tuple(dirichlet_tags))

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
        
    a = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx -inner(p, div(v))*dx
         -inner(q, div(u))*dx)
        
    L = inner(f, v)*dx
    L += sum(inner(dot(sigma, n), v)*ds(tag) for tag in neumann_tags)

    return a, L, bcs, W

# --------------------------------------------------------------------

if __name__ == '__main__':
    from firedrake.petsc import PETSc
    import numpy as np

    print = PETSc.Sys.Print
    
    OptDB = PETSc.Options()
    # System size    
    n0 = OptDB.getInt('ncells', 1)
    nrefs = OptDB.getInt('nrefs', 2)
    mu = OptDB.getScalar('mu', 1E0)
    L_ = OptDB.getScalar('L', 1E0)

    pc_type = OptDB.getString('pc_type', 'mg')

    dtags = (1, 2, 3)

    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

    # Customize solver
    solver_parameters = {
        'ksp_converged_reason': None,
        'ksp_view_eigenvalues': None,
        # 'ksp_view_singularvalues': None,
        'ksp_monitor_true_residual': None,
        "ksp_monitor": None,
        'ksp_view': None,
	# 'ksp_monitor_singular_value': None,
        #
        'ksp_rtol': 1E-10,
        "ksp_type": "gmres",
    }

    if pc_type == 'mg':
        pc_parameters = {
            "pc_type": "mg",
            "pc_mg_type": 'multiplicative', #"full",
            'pc_mg_cycle_type': 'v',
            'pc_mg_view': None,
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_max_it": 2,
            "mg_levels_pc_type": "python",
            "mg_levels_pc_python_type": "firedrake.PatchPC",
            "mg_levels_patch_pc_patch_local_type": "additive",
            "mg_levels_patch_pc_patch_partition_of_unity": False,
            "mg_levels_patch_pc_patch_construct_type": "vanka",
            "mg_levels_patch_pc_patch_construct_dim": 0,
            "mg_levels_patch_pc_patch_exclude_subspaces": "1",
        }
    else:
        pc_parameters = {
            "mat_type": "matfree",
            #
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "pc_fieldsplit_schur_fact_type": "diag",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "python",
            "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
            "fieldsplit_0_assembled_pc_type": "lu",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "python",
            "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
            "fieldsplit_1_Mp_ksp_type": "preonly",
            "fieldsplit_1_Mp_pc_type": "lu"
        }

    solver_parameters.update(pc_parameters)

    iters_history = []
    pressure_kernel = set(dtags) == {1, 2, 3, 4}

    h0, errors0 = None, None
    n = 2**n0
    for i in range(nrefs):
        coarse_mesh = RectangleMesh(n, int(L_)*n, 1, L_, distribution_parameters=distribution_parameters)
        hierarchy = MeshHierarchy(coarse_mesh, 3)    
        mesh = hierarchy[-1]

        mms = stokes_mms(mesh, params={'mu': mu, 'L': L_})
        a, L, bcs, W  = stokes_system(mesh, mms=mms, dirichlet_tags=dtags)        
        
        wh = Function(W)

        if pressure_kernel:
            nullspace = MixedVectorSpaceBasis(
                W, [W.sub(0), VectorSpaceBasis(constant=True)])
        else:
            nullspace = None

        problem = LinearVariationalProblem(a, L, wh, bcs=bcs)
        solver = LinearVariationalSolver(problem, nullspace=nullspace, solver_parameters=solver_parameters)
        solver.solve()

        ksp = solver.snes.ksp
        niters = ksp.getIterationNumber()

        lmin, lmax = np.sort(np.abs(ksp.computeEigenvalues()))[[0, -1]]
        cond = lmax/lmin
        
        dofs = W.dim()

        errors = mms['get_errors'](wh, mean=pressure_kernel)
        msg = ' '.join([f'{key} = {val:.3E}' for key, val in errors.items()])
        msg = ' '.join([msg, f'niters = {niters}', f'dofs = {dofs}'])
        msg = ' '.join([msg, f'lmin = {lmin:.3E}', f'lmax = {lmax:.3E}', f'cond = {cond:.2f}'])        
        print(GREEN % msg)

        iters_history.append((dofs, niters))

        n *= 2
    # Done
    np.savetxt(f'itersHistory_L{L_}_pcType{pc_type}.txt', np.array(iters_history))

    uh, ph = wh.split()
    uh.rename("Velocity")
    ph.rename("Pressure")

    File(f"stokes_L{L_}_pcType{pc_type}.pvd").write(uh, ph)        
