import numpy as np
import lsdo_function_spaces as lfs
from joblib import Parallel, delayed
import pandas as pd
import re


def import_file(file_name:str, parallelize:bool=True) -> lfs.Function:
    ''' Read file '''
    with open(file_name, 'r') as f:
        print('Importing OpenVSP file:', file_name)
        if 'B_SPLINE_SURFACE_WITH_KNOTS' not in f.read():
            print("No surfaces found!!")
            print("Something is wrong with the file" \
                , "or this reader doesn't work for this file.")
            return

    '''Stage 1: Parse all information and line numbers for each surface and create B-spline objects'''
    b_splines = {}
    b_spline_spaces = {}
    b_splines_to_spaces_dict = {}
    parsed_info_dict = {}
    with open(file_name, 'r') as f:
        b_spline_surf_info = re.findall(r"B_SPLINE_SURFACE_WITH_KNOTS.*\)", f.read())
    num_surf = len(b_spline_surf_info)

    if parallelize:
        parsed_info_tuples = Parallel(n_jobs=20)(delayed(_parse_file_info)(i, surf) for i, surf in enumerate(b_spline_surf_info))
    else:
        parsed_info_tuples = []
        for i, surf in enumerate(b_spline_surf_info):
            parsed_info_tuples.append(_parse_file_info(i, surf))

    for i in range(num_surf):
        parsed_info_tuple = parsed_info_tuples[i]
        parsed_info = parsed_info_tuple[0]

        space_name = parsed_info_tuple[1]
        if space_name in b_spline_spaces:
            b_spline_space = b_spline_spaces[space_name]
        else:
            knots_u = np.array([parsed_info[10]])
            knots_u = np.repeat(knots_u, parsed_info[8])
            knots_u = knots_u/knots_u[-1]
            knots_v = np.array([parsed_info[11]])
            knots_v = np.repeat(knots_v, parsed_info[9])
            knots_v = knots_v/knots_v[-1]

            order_u = int(parsed_info[1])+1
            order_v = int(parsed_info[2])+1

            coefficients_shape = tuple([len(knots_u)-order_u, len(knots_v)-order_v])
            knots = np.hstack((knots_u, knots_v))
            b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(order_u-1, order_v-1),
                                              coefficients_shape=coefficients_shape + (3,))
            # NOTE: # Hardcoding 3 for num_physical_dimensions because this import is hardcoded for OpenVSP anyway
            b_spline_spaces[space_name] = b_spline_space

        b_splines_to_spaces_dict[parsed_info[0]] = space_name

        parsed_info_dict[f'surf{i}_name'] = parsed_info[0]
        parsed_info_dict[f'surf{i}_cp_line_nums'] = np.array(parsed_info[3])
        parsed_info_dict[f'surf{i}_u_multiplicities'] = np.array(parsed_info[8])
        parsed_info_dict[f'surf{i}_v_multiplicities'] = np.array(parsed_info[9])


    ''' Stage 2: Replace line numbers of control points with control points arrays'''
    line_numbs_total_array = np.array([])
    for i in range(num_surf):
        line_numbs_total_array = np.append(line_numbs_total_array, parsed_info_dict[f'surf{i}_cp_line_nums'].flatten())
    point_table = pd.read_csv(file_name, sep='=', names=['lines', 'raw_point'])
    filtered_point_table = point_table.loc[point_table["lines"].isin(line_numbs_total_array)]
    point_table = pd.DataFrame(filtered_point_table['raw_point'].str.findall(r"(-?\d+\.\d*E?-?\d*)").to_list(), columns=['x', 'y', 'z'])
    point_table["lines"] = filtered_point_table["lines"].values

    if parallelize:
        b_spline_list = Parallel(n_jobs=20)(delayed(_build_b_splines)(i, parsed_info_dict, point_table, b_spline_spaces, 
                                                                      b_splines_to_spaces_dict) for i in range(num_surf))
    else:
        b_spline_list = []
        for i in range(num_surf):
            b_spline_list.append(_build_b_splines(i, parsed_info_dict, point_table, b_spline_spaces, b_splines_to_spaces_dict))

    b_splines = {}
    num_parametric_dimensions = {}
    index_to_space = {}
    name_to_index = {}
    coefficients = []
    for i, b_spline in enumerate(b_spline_list):
        b_splines[b_spline.name] = b_spline
        num_parametric_dimensions[i] = 2
        name_to_index[b_spline.name] = i
        coefficients.append(b_spline.coefficients.value.reshape((b_spline.coefficients.size//3,3)))
        index_to_space[i] = list(b_spline_spaces.keys()).index(b_splines_to_spaces_dict[b_spline.name])
    coefficients = np.vstack(coefficients)

    b_spline_spaces = list(b_spline_spaces.values())
    b_spline_set_space = lfs.FunctionSetSpace(num_parametric_dimensions=num_parametric_dimensions, spaces=b_spline_spaces,
                                              index_to_space=index_to_space, name_to_index=name_to_index)
    b_spline_set = lfs.Function(space=b_spline_set_space, coefficients=coefficients)
        
    print('Complete import')
    return b_spline_set
    # return b_splines

def _parse_file_info(i, surf):
    #print(surf)
        # Get numbers following hashes in lines with B_SPLINE... These numbers should only be the line numbers of the cntrl_pts
        info_index = 0
        parsed_info = []
        while(info_index < len(surf)):
            if(surf[info_index]=="("):
                info_index += 1
                level_1_array = []
                while(surf[info_index]!=")"):
                    if(surf[info_index]=="("):
                        info_index += 1
                        level_2_array = []

                        while(surf[info_index]!=")"):
                            if(surf[info_index]=="("):
                                info_index += 1
                                nest_level3_start_index = info_index
                                level_3_array = []
                                while(surf[info_index]!=")"):
                                    info_index += 1
                                level_3_array = surf[nest_level3_start_index:info_index].split(', ')
                                level_2_array.append(level_3_array)
                                info_index += 1
                            else:
                                level_2_array.append(surf[info_index])
                                info_index += 1
                        level_1_array.append(level_2_array)
                        info_index += 1
                    elif(surf[info_index]=="'"):
                        info_index += 1
                        level_2_array = []
                        while(surf[info_index]!="'"):
                            level_2_array.append(surf[info_index])
                            info_index += 1
                        level_2_array = ''.join(level_2_array)
                        level_1_array.append(level_2_array)
                        info_index += 1
                    else:
                        level_1_array.append(surf[info_index])
                        info_index += 1
                info_index += 1
            else:
                info_index += 1
        info_index = 0
        last_comma = 1
        while(info_index < len(level_1_array)):
            if(level_1_array[info_index]==","):
                if(((info_index-1) - last_comma) > 1):
                    parsed_info.append(''.join(level_1_array[(last_comma+1):info_index]))
                else:
                    parsed_info.append(level_1_array[info_index-1])
                last_comma = info_index
            elif(info_index==(len(level_1_array)-1)):
                parsed_info.append(''.join(level_1_array[(last_comma+1):(info_index+1)]))
            info_index += 1

        while "," in parsed_info[3]:
            parsed_info[3].remove(',')
        for j in range(4):
            parsed_info[j+8] = re.findall('\d+' , ''.join(parsed_info[j+8]))
            if j <= 1:
                info_index = 0
                for ele in parsed_info[j+8]:
                    parsed_info[j+8][info_index] = int(ele)
                    info_index += 1
            else:
                info_index = 0
                for ele in parsed_info[j+8]:
                    parsed_info[j+8][info_index] = float(ele)
                    info_index += 1

        parsed_info[0] = parsed_info[0][17:]+f', {i}'   # Hardcoded 17 to remove useless string

        knots_u = np.array([parsed_info[10]])
        knots_u = np.repeat(knots_u, parsed_info[8])
        knots_u = knots_u/knots_u[-1]
        knots_v = np.array([parsed_info[11]])
        knots_v = np.repeat(knots_v, parsed_info[9])
        knots_v = knots_v/knots_v[-1]

        order_u = int(parsed_info[1])+1
        order_v = int(parsed_info[2])+1
        space_name = 'order_u_' + str(order_u) + 'order_v_' + str(order_v) + 'knots_u_' + str(knots_u) + 'knots_v_' + str(knots_v) + \
            '_b_spline_space'

        return (parsed_info, space_name)


def _build_b_splines(i, parsed_info_dict, point_table, b_spline_spaces, b_splines_to_spaces_dict):
    num_rows_of_cps = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[0]
    num_cp_per_row = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[1]
    cntrl_pts = np.zeros((num_rows_of_cps, num_cp_per_row, 3))
    for j in range(num_rows_of_cps):
        col_cntrl_pts = point_table.loc[point_table["lines"].isin(parsed_info_dict[f'surf{i}_cp_line_nums'][j])][['x', 'y', 'z']]
        if ((len(col_cntrl_pts) != num_cp_per_row) and (len(col_cntrl_pts) != 1)):
            for k in range(num_cp_per_row):
                cntrl_pts[j,k,:] = point_table.loc[point_table["lines"]==parsed_info_dict[f'surf{i}_cp_line_nums'][j][k]][['x', 'y', 'z']]
        else:
            cntrl_pts[j,:,:] = col_cntrl_pts

    # u_multiplicities = parsed_info_dict[f'surf{i}_u_multiplicities']
    # v_multiplicities = parsed_info_dict[f'surf{i}_v_multiplicities']
    # control_points_shape = (np.sum(u_multiplicities), np.sum(v_multiplicities), 3)
    # control_points_with_multiplicity = np.zeros(control_points_shape)
    # u_counter = 0
    # v_counter = 0
    # for j in range(num_rows_of_cps):
    #     u_counter_end = u_counter + u_multiplicities[j]
    #     for k in range(num_cp_per_row):
    #         v_counter_end = v_counter + v_multiplicities[k]
    #         control_points_with_multiplicity[u_counter:u_counter_end, v_counter:v_counter_end, :] = cntrl_pts[j,k,:]

    b_spline_name = parsed_info_dict[f'surf{i}_name']
    coefficients = cntrl_pts.reshape((-1,))
    b_spline_space = b_spline_spaces[b_splines_to_spaces_dict[b_spline_name]]
    num_physical_dimensions = cntrl_pts.shape[-1]
    coefficients = coefficients.reshape((b_spline_space.coefficients_shape[0], b_spline_space.coefficients_shape[1], num_physical_dimensions))
    # print(f'Creating B-spline {b_spline_name}')
    # import csdl_alpha as csdl
    # print(csdl.manager)
    # b_spline = lfs.Function(name=b_spline_name, space=b_spline_space, coefficients=coefficients, num_physical_dimensions=num_physical_dimensions)
    b_spline = lfs.Function(space=b_spline_space, coefficients=coefficients)
    b_spline.name = b_spline_name
    return b_spline