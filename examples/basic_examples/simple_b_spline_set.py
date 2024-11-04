import lsdo_function_spaces as lfs
import csdl_alpha as csdl
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()

file_path = 'examples/import_files_for_examples/'
file_name = 'rectangular_wing.stp'
wing = lfs.import_file(file_path + file_name, parallelize=False)

# Plot the wing
wing.plot()

fs = lfs.RBFFunctionSpace(num_parametric_dimensions=2, radial_function='polyharmonic_spline', grid_size=(20,20))
new_wing = wing.refit(fs, grid_resolution=(100,100))
new_wing.plot()
exit()

wing_up = wing/2
wing_up.plot()

exit()


# left_wing = wing.create_subset(function_search_names=[', 0'])
# left_wing.plot()

# left_wing.functions[3].coefficients += 1.
# left_wing.plot()

plotting_elements = wing.plot(show=False)
wing.plot(point_types=['coefficients'], plot_types=['point_cloud'], color='#C69214', additional_plotting_elements=plotting_elements)


# wing.project(np.array([0., 0., 0.]))
new_function_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1,3), coefficients_shape=(10,30))
new_function_spaces = [new_function_space]*len(wing.functions)
new_wing = wing.refit(new_function_spaces=new_function_spaces, grid_resolution=(30,100))
plotting_elements = new_wing.plot(show=False)
new_wing.plot(point_types=['coefficients'], plot_types=['point_cloud'], color='#C69214', additional_plotting_elements=plotting_elements)
print('hi')