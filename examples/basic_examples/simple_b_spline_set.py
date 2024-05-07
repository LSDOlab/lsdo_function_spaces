import lsdo_function_spaces as lfs
import csdl_alpha as csdl


recorder = csdl.Recorder(inline=True)
recorder.start()

file_path = 'examples/import_files_for_examples/'
file_name = 'rectangular_wing.stp'
wing = lfs.import_file(file_path + file_name, parallelize=False)

# Plot the wing
wing.plot()
print('hi')