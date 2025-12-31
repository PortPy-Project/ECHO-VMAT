"""

"""
import os.path
import time
import numpy as np
import portpy.photon as pp
from echo_vmat.echo_vmat_optimization import EchoVmatOptimization
from echo_vmat.echo_vmat_optimization_col_gen import EchoVmatOptimizationColGen
from echo_vmat.utils.get_sparse_only import get_sparse_only
import matplotlib.pyplot as plt
from echo_vmat.arcs import Arcs
from copy import deepcopy
import pandas as pd
import sys
import json


def echo_vmat_portpy():
    """
     1) Accessing the portpy data (DataExplorer class)

     To start using this resource, users are required to download the latest version of the dataset, which can be found at (https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit). Then, the dataset can be accessed as demonstrated below.

    """

    tic_all = time.time()
    data_dir = r'../../PortPy/data'
    data = pp.DataExplorer(data_dir=data_dir)
    data.patient_id = 'Lung_Phantom_Patient_1'

    # Load clinical criteri and optimization parameters from the config files. They are located in ./echo_vmat/config_files. Users can modify the clinical criteria and optimization parameters based upon their needs
    config_path = os.path.join(os.getcwd(), '..', 'echo_vmat', 'config_files')
    protocol_name = 'Lung_2Gy_30Fx'
    filename = os.path.join(config_path, protocol_name + '_opt_params.json')
    vmat_opt_params = data.load_json(file_name=filename)

    filename = os.path.join(config_path, protocol_name + '_clinical_criteria.json')
    clinical_criteria = pp.ClinicalCriteria(file_name=filename)

    if "flag_full_matrix" in vmat_opt_params['opt_parameters']:
        flag_full_matrix = vmat_opt_params['opt_parameters']['flag_full_matrix']
    else:
        flag_full_matrix = False
    structs = pp.Structures(data)
    # Users can modify the arcs based upon the start and stop gantry angles.
    # Below is an example where 2 arcs area created,
    # the start and stop gantry angles 0 to 180 and 180 to 359 are used.
    all_beam_ids = np.arange(0, 37)  # 72 beams with gantry angle between 0 to 359 deg and spacing 5 deg
    arcs_dict = {'arcs': [{'arc_id': "01", "beam_ids": all_beam_ids[0:int(len(all_beam_ids) / 2)]},
                          {'arc_id': "02", "beam_ids": all_beam_ids[int(len(all_beam_ids) / 2):]}]}

    beam_ids = [beam_id for arcs_dict in arcs_dict['arcs'] for beam_id in arcs_dict['beam_ids']]
    beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=flag_full_matrix)

    if 'Patient Surface' in structs.get_structures():
        ind = structs.structures_dict['name'].index('Patient Surface')
        structs.structures_dict['name'][ind] = 'BODY'

    # temporary capitalize the structures
    for i in range(len(structs.structures_dict['name'])):
        structs.structures_dict['name'][i] = structs.structures_dict['name'][i].upper()

    # # Creating optimization structures (i.e., Rinds, PTV-GTV)
    for i in range(len(vmat_opt_params["steps"])):
        structs.create_opt_structures(opt_params=vmat_opt_params["steps"][str(i + 1)],
                                      clinical_criteria=clinical_criteria)

    inf_matrix = pp.InfluenceMatrix(structs=structs, beams=beams, is_full=flag_full_matrix)  # is_bev=True

    # use naive or RMR sparsification
    if flag_full_matrix:
        A = deepcopy(inf_matrix.A)
        # A = A * np.float32(2 * clinical_criteria.get_prescription() / clinical_criteria.get_num_of_fractions())
        if "threshold_perc" in vmat_opt_params['opt_parameters']:
            threshold_perc = vmat_opt_params['opt_parameters']['threshold_perc']
        else:
            threshold_perc = 5
        if "sparsification" in vmat_opt_params['opt_parameters']: # Default is RMR sparsification for more accurate results. It can be changed to Naive
            sparsification = vmat_opt_params['opt_parameters']['sparsification']
        else:
            sparsification = 'Naive'
        # threshold_abs = 0.0112*2*clinical_criteria.get_prescription() / clinical_criteria.get_num_of_fractions()
        B = get_sparse_only(A, threshold_perc=threshold_perc, compression=sparsification)
        # # calculate delta
        # delta = A.sum(axis=1) - B.sum(axis=1).A1
        # inf_matrix.opt_voxels_dict['delta'][0] = delta
        inf_matrix.A = B

    # scale influence matrix to get correct MU for TPS import
    if "inf_matrix_scale_factor" in vmat_opt_params['opt_parameters']:
        inf_matrix_scale_factor = vmat_opt_params['opt_parameters']['inf_matrix_scale_factor']
    else:
        inf_matrix_scale_factor = 1
    print("inf_matrix_scale_factor: ", inf_matrix_scale_factor)

    inf_matrix.A = inf_matrix.A * np.float32(inf_matrix_scale_factor)

    arcs = Arcs(arcs_dict=arcs_dict, inf_matrix=inf_matrix)

    if 'voxel_coordinate_XYZ_mm' not in inf_matrix.opt_voxels_dict:
        inf_matrix.opt_voxels_dict['voxel_coordinate_XYZ_mm'] = [None]
        inf_matrix.opt_voxels_dict['voxel_coordinate_XYZ_mm'][0] = inf_matrix.get_voxel_coordinates()
    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(structs=structs,
                      beams=beams,
                      inf_matrix=inf_matrix,
                      clinical_criteria=clinical_criteria,
                      arcs=arcs)

    # generate initial leaf positions using column generation approach else use user defined initial leaf positions
    if vmat_opt_params['opt_parameters']['initial_leaf_pos'].lower() == 'cg':
        start_col_gen = time.time()
        vmat_opt = EchoVmatOptimizationColGen(my_plan=my_plan,
                                              opt_params=vmat_opt_params,
                                              step_num=1)

        sol_col_gen = vmat_opt.run_col_gen_algo(solver='MOSEK', verbose=True, accept_unknown=True)
        dose_1d = inf_matrix.A @ sol_col_gen['optimal_intensity'] * my_plan.get_num_of_fractions()
        # # plot dvh for the above structures
        fig, ax = plt.subplots(figsize=(12, 8))
        struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'RIND_0', 'RIND_1', 'LUNGS_NOT_GTV', 'RECT_WALL', 'BLAD_WALL',
                        'URETHRA']
        ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d,
                                       struct_names=struct_names, ax=ax)
        ax.set_title('Initial Col gen dvh')
        plt.show(block=False)
        plt.close('all')
        # pp.save_optimal_sol(sol=sol_col_gen, sol_name='sol_col_gen', path=r'C:\Temp')

        end_col_gen = time.time()
        print('***************time to generate initial leaf positions = ', end_col_gen - start_col_gen, 'seconds *****************')

    # check if there are dvh constraints and run step 0
    clinical_criteria.get_dvh_table(my_plan=my_plan, opt_params=vmat_opt_params['steps']['2'])
    vmat_opt = EchoVmatOptimization(my_plan=my_plan,
                                    opt_params=vmat_opt_params,
                                    step_num=1)
    solutions = []
    sol = {}
    final_convergence = []
    # Run step 0 for dvh optimization
    if not clinical_criteria.dvh_table.empty:
        sol_convergence = vmat_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=True)
        final_convergence.extend(sol_convergence)
        sol = sol_convergence[vmat_opt.best_iteration]
        solutions.append(sol)
        vmat_opt.update_params(step_number=0, sol=sol)

    # Run step 1 and 2 for hierarchical optimization
    for i in range(2):
        step_time = time.time()
        if not clinical_criteria.dvh_table.empty:
            dose = sol['act_dose_v'] * my_plan.get_num_of_fractions()
            clinical_criteria.get_low_dose_vox_ind(my_plan, dose=dose)  # get low dose voxels and update dvh table
        vmat_opt.set_step_num(i + 1)
        # running scp algorithm
        sol_convergence = vmat_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=True)
        final_convergence.extend(sol_convergence)
        sol = final_convergence[vmat_opt.best_iteration]
        solutions.append(sol)
        vmat_opt.update_params(step_number=i + 1, sol=sol)
        print('***************Time for step {}:{} *******************'.format(i + 1, time.time() - step_time))

    # # plot dvh for the above structures
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'RIND_0', 'RIND_1', 'LUNGS_NOT_GTV', 'RECT_WALL', 'BLAD_WALL',
                    'URETHRA','LUNG_L', 'LUNG_R']
    title = []
    style = ['-', '--', ':']
    for i in range(len(solutions)):
        ax = pp.Visualization.plot_dvh(my_plan, dose_1d=solutions[i]['act_dose_v'] * my_plan.get_num_of_fractions(),
                                       struct_names=struct_names,
                                       style=style[i], ax=ax)
        if len(solutions) < 3:
            title.append(f'Step {i + 1} {style[i]}')
        else:
            title.append(f'Step {i} {style[i]}')
    ax.set_title(" ".join(title))
    plt.show(block=False)
    plt.close('all')

    print('saving optimal solution..')
    for i in range(len(solutions)):
        if len(solutions) < 3:
            sol_name = f'sol_step{i + 1}.pkl'
        else:
            sol_name = f'sol_step{i}.pkl'
        pp.save_optimal_sol(sol=solutions[i], sol_name=sol_name, path=os.path.join('C', 'Temp', data.patient_id))

    print('saving my_plan..')
    pp.save_plan(my_plan, 'my_plan.pkl', path=os.path.join('C', 'Temp', data.patient_id))

    # update done
    opt_time = round(time.time() - tic_all, 2)
    print('***************** opt_time (secs) ********************:', opt_time)


if __name__ == "__main__":
    echo_vmat_portpy()
