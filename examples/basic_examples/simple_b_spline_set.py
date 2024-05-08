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

left_wing = wing.create_subset(function_search_names=[', 0'])
left_wing.plot()

# wing.project(np.array([0., 0., 0.]))

print('hi')