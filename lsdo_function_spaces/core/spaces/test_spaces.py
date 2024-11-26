from lsdo_function_spaces.core.spaces.b_spline_space import *
from lsdo_function_spaces.core.spaces.idw_space import *
from lsdo_function_spaces.core.spaces.tri_space import *

from scipy.stats.qmc import LatinHypercube


def test_fit_eval():
    import lsdo_function_spaces as lfs
    cases = []

    # B-Spline
    cases.append({
        'space': lfs.BSplineSpace(num_parametric_dimensions=2, degree=(2,2), coefficients_shape=(10,10)),
        'tol': 1e-4
    })

    # IDW
    cases.append({
        'space': lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, conserve=False, grid_size=100),
        'tol': 1e-3
    })
    cases.append({
        'space': lfs.IDWFunctionSpace(num_parametric_dimensions=2, order=4, conserve=False, grid_size=100, n_neighbors=10),
        'tol': 1e-3
    })

    # RBF
    cases.append({
        'space': lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function='gaussian', grid_size=(10,10), epsilon=2),
        'tol': 1e-4
    })
    cases.append({
        'space': lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function='polyharmonic_spline', grid_size=(10,10), k=2),
        'tol': 1e-3
    })
    cases.append({
        'space': lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function='inverse_quadratic', grid_size=(10,10), epsilon=2),
        'tol': 1e-4
    })
    cases.append({
        'space': lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function='inverse_multiquadric', grid_size=(10,10), epsilon=2),
        'tol': 1e-4
    })
    cases.append({
        'space': lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function='bump', grid_size=(10,10), epsilon=1/1.6),
        'tol': 1e-3
    })

    # Polynomial
    cases.append({
        'space': lfs.PolynomialSpace(num_parametric_dimensions=2, order=7),
        'tol': 1e-4
    })

    # Linear Triangulation
    cases.append({
        'space': lfs.LinearTriangulationSpace(grid_size=(10,10)),
        'tol': 1e-3
    })


    for case in cases:
        rec = csdl.Recorder(inline=True)
        rec.start()
        
        space:lfs.FunctionSpace = case['space']
        tol = case['tol']

        np.random.seed(0)
        num_points = 1000
        parametric_coordinates = LatinHypercube(d=2, seed=7).random(num_points)
        height = (np.sin(2*np.pi*parametric_coordinates[:,0]) + .5*np.cos(2*np.pi*parametric_coordinates[:,1])).reshape(-1,1)
        data = np.hstack((parametric_coordinates*10, height))

        function = space.fit_function(data, parametric_coordinates)
        # function.project(data, plot=True, do_pickles=False)
        # function.plot()
        eval_data = function.evaluate(parametric_coordinates)

        if not np.linalg.norm(eval_data.value - data)/num_points < tol:
            print(f'error is {np.linalg.norm(eval_data.value - data)/num_points}')
            raise ValueError('Failed basic test for space: {}'.format(space.__class__.__name__))
        

if __name__ == "__main__":
    # test_tri()
    test_fit_eval()