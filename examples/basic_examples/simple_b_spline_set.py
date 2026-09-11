import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np
from lsdo_function_spaces import operations as ops

recorder = csdl.Recorder(inline=True)
recorder.start()

file_path = 'examples/import_files_for_examples/'
# file_name = 'lift_plus_cruise_fuse_wing_tail.stp'
file_name = 'rectangular_wing.stp'
wing = lfs.import_file(file_path + file_name, parallelize=False)
wing_ctrl_pts = []
for fun_ind, fun in wing.functions.items():
    wing_ctrl_pts.append(fun.coefficients.value.reshape(-1, 3))
wing_ctrl_pts = np.vstack(wing_ctrl_pts)
ctrl_pt_plot = lfs.plot_points(wing_ctrl_pts, color='red', show=False)
wing.plot(additional_plotting_elements=[ctrl_pt_plot], opacity=0.7)
print("Using canonical BSplineSpace:", isinstance(wing.functions[0].space, lfs.BSplineSpace))
exit()