from __future__ import annotations

from portpy.photon import Optimization
from typing import TYPE_CHECKING
import time
from echo_vmat.utils.get_apt_reg_metric import get_apt_reg_metric

if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
from portpy.photon.clinical_criteria import ClinicalCriteria
import cvxpy as cp
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
from scipy.spatial import cKDTree

from scipy.ndimage import binary_erosion, label
from sklearn.neighbors import NearestNeighbors
import json


class EchoVmatOptimizationColGen(Optimization):
    def __init__(self, my_plan: Plan, arcs=None, inf_matrix: InfluenceMatrix = None,
                 clinical_criteria: ClinicalCriteria = None,
                 opt_params: dict = None, vars: dict = None, sol=None, step_num: int = None,
                 is_corr: bool = False, delta: np.ndarray = None):

        # combine steps objective functions into one
        if 'col_gen' in opt_params:
            super().__init__(my_plan=my_plan, inf_matrix=inf_matrix,
                             clinical_criteria=clinical_criteria,
                             opt_params=opt_params["col_gen"], vars=vars)
            opt_params = opt_params["col_gen"]  # col_gen is the new opt parameters dictionary for col gen
        else:
            if len(opt_params['steps']) > 1:
                opt_params["steps"][str(step_num)]["objective_functions"] = self.remove_duplicates(
                    opt_params["steps"][str(step_num)]["objective_functions"] + opt_params["steps"][str(step_num + 1)]["objective_functions"])
                opt_params["steps"][str(step_num)]["constraints"] = self.remove_duplicates(opt_params["steps"][str(step_num)]["constraints"] + opt_params["steps"][str(step_num + 1)]["constraints"])
            # Call the constructor of the base class (Optimization) using super()
            super().__init__(my_plan=my_plan, inf_matrix=inf_matrix,
                             clinical_criteria=clinical_criteria,
                             opt_params=opt_params["steps"][str(step_num)], vars=vars)
        # save previous solution if passed

        self.prev_sol = sol
        self.distance_from_structure = {}
        self.step_num = step_num
        if arcs is None:
            self.arcs = my_plan.arcs
        else:
            self.arcs = arcs
        self.cvxpy_params = {}
        self.vmat_params = opt_params['opt_parameters']
        self.all_params = opt_params
        self.obj_funcs = None
        self.constraint_def = None
        self.outer_iteration = 0
        self.best_iteration = None
        self.obj_actual = []
        self.constraints_actual = []
        self.is_corr = is_corr
        self.constraints_ind_map = {}
        if delta is None:
            self.delta = np.zeros(self.inf_matrix.A.shape[0])

    def calculate_scores_grad(self, A, A_indices, sol):
        """
        Efficiently calculate scores for unselected beamlets using gradients.

        Parameters:
            A (csr_matrix): Influence matrix (voxels x beamlets).
            A_indices (list): List of indices of selected beamlets.
            sol (dict): Dictionary containing solution intensities ('y').

        Returns:
            list: List of tuples (beamlet index, score) sorted by score (lower is better).
        """
        # Step 1: Reconstruct B matrix from A_indices
        if not A_indices:  # Empty B
            self.B = np.empty((A.shape[0], 0))

        # Flatten A_indices and extract columns from A
        selected_beamlets = [item for sublist in A_indices for item in sublist]
        # Get unselected beamlets index
        selected_indices_set = set(selected_beamlets)
        unselected_mask = np.array([i not in selected_indices_set for i in range(A.shape[1])])

        # Step 2: Compute current dose with selected apertures
        current_dose = self.B @ sol['y'] if self.B.shape[1] > 0 else np.zeros(A.shape[0])

        # Initialize gradients
        num_beamlets = A.shape[1]
        overdose_grad = np.zeros(num_beamlets)
        underdose_grad = np.zeros(num_beamlets)
        quadratic_grad = np.zeros(num_beamlets)
        constraints_grad = np.zeros(num_beamlets)

        # Objective function gradients
        obj_funcs = self.obj_funcs
        structures = self.my_plan.structures
        inf_matrix = self.my_plan.inf_matrix
        num_fractions = self.my_plan.get_num_of_fractions()

        for obj in obj_funcs:
            if obj['type'] == 'quadratic-overdose':
                struct = obj['structure_name']
                if struct not in structures.get_structures():
                    continue
                voxels = inf_matrix.get_opt_voxels_idx(struct)
                if len(voxels) == 0:
                    continue
                dose_gy = self.get_dose_gy_per_frac_for_opt(obj, struct=struct)
                influence = A[voxels, :]
                overdose = current_dose[voxels] - dose_gy
                overdose[overdose < 0] = 0
                overdose_grad += (2 * obj['weight'] / len(voxels)) * (influence.T @ overdose)
            elif obj['type'] == 'quadratic-underdose':
                struct = obj['structure_name']
                if struct not in structures.get_structures():
                    continue
                voxels = inf_matrix.get_opt_voxels_idx(struct)
                if len(voxels) == 0:
                    continue
                dose_gy = self.get_dose_gy_per_frac_for_opt(obj, struct=struct)
                influence = A[voxels, :]
                underdose = current_dose[voxels] - dose_gy
                underdose[underdose > 0] = 0
                underdose_grad += (2 * obj['weight'] / len(voxels)) * (influence.T @ underdose)
            elif obj['type'] == 'quadratic':
                struct = obj['structure_name']
                if struct not in structures.get_structures():
                    continue
                voxels = inf_matrix.get_opt_voxels_idx(struct)
                if len(voxels) == 0:
                    continue
                influence = A[voxels, :]
                quadratic_grad += (self.vmat_params['normalize_oar_weights'] * 2 * obj['weight'] / len(voxels)) * (influence.T @ current_dose[voxels])

        # Constraints gradients
        constraint_def = self.constraint_def
        constraints_ind_map = self.constraints_ind_map

        if constraints_ind_map:
            for constraint in constraint_def:

                if constraint['type'] == 'max_dose':
                    struct = constraint['parameters']['structure_name']
                    if struct not in structures.get_structures():
                        continue
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    if len(voxels) == 0:
                        continue
                    limit_key = self.matching_keys(constraint['constraints'], 'limit')
                    if limit_key:
                        limit = self.dose_to_gy(limit_key, constraint['constraints'][limit_key]) / num_fractions
                        dual = self.constraints[constraints_ind_map['max_dose_' + struct + str(limit)]].dual_value
                        constraints_grad += np.squeeze(np.asarray(dual * (A[voxels, :])))

                if constraint['type'] == 'DFO':
                    dfo, oar_voxels = self.get_dfo_parameters(dfo_dict=constraint, is_obj=False)
                    dual = self.constraints[constraints_ind_map['max_dfo']].dual_value
                    constraints_grad += np.squeeze(np.asarray(dual * (A[oar_voxels, :])))

        # Combine all gradients to calculate final scores
        scores = (overdose_grad + underdose_grad + quadratic_grad + constraints_grad)
        scores = scores / np.max(abs(scores))
        # scores = scores/np.max(abs(scores[unselected_mask]))  # normalize only using unselected beamlets
        return scores

    def select_best_aperture(self, remaining_beam_ids, scores, smooth_delta=0.1, smooth_beta=5, apt_reg_type_cg='row-by-row'):
        """

        :param remaining_beam_ids: remaining beam ids from which aperture should be selected
        :param scores: scores for unselected beamlets
        :param smooth_delta: smoothness weight to modify the score based on leaf positions
        :param apt_reg_type_cg: strategy to smooth the leafs based on their position. Default to row-by-row. Can be changed to greedy

        :return:best beam id and the beamlets selected for that beam
        """
        aperture_score = {}
        selected_beamlets = {}
        selected_leafs = {}
        # Map beamlet indices to their scores for quick lookup
        if isinstance(scores, np.ndarray):
            scores = list(scores)
            score_dict = {index: score for index, score in enumerate(scores)}
        else:
            score_dict = {index: score for index, score in scores}
        # score_dict_orig = score_dict.copy()
        for arc in self.arcs.arcs_dict['arcs']:
            for beam in arc['vmat_opt']:
                if beam['beam_id'] in remaining_beam_ids:
                    cumulative_score = 0
                    selected_beamlets[beam['beam_id']] = []
                    selected_leafs[beam['beam_id']] = [[] for _ in range(beam['num_rows'])]
                    remaining_rows = list(range(beam['num_rows']))  # Start with all rows
                    processed_rows = []
                    r_best = -1
                    while remaining_rows:

                        # identify the best row
                        # get scores for each remaining row

                        if apt_reg_type_cg == 'greedy':
                            row_scores = [sum(score_dict[beamlet] for beamlet in row[row >= 0]) for r, row in enumerate(beam['reduced_2d_grid']) if r in remaining_rows]
                            r_best = np.argmin(row_scores)
                            row = beam['reduced_2d_grid'][remaining_rows[r_best]]
                        else:
                            r_best += 1
                            row = beam['reduced_2d_grid'][r_best]

                        # scores_matrix = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                        # Step 0: Initialization
                        c1, c2 = np.argwhere(row >= 0)[0][0], np.argwhere(row >= 0)[0][0] - 1  # start with one index less for c2
                        c1_opt, c2_opt = c1, c2 + 1
                        v, v_bar, v_star = 0, 0, 0  # Initialize v, v_bar, and v_star

                        # Number of beamlets in the row
                        n = np.argwhere(row >= 0)[-1][0]

                        # Step 1: Iterate over possible right leaf positions
                        while c2 < n:
                            # Calculate v incrementally by adding beamlet scores in the range [c1, c2]
                            v += score_dict.get(row[c2 + 1])

                            # Step 2: Update v_bar if necessary
                            if v > v_bar:
                                v_bar = v
                                c1 = c2 + 2  # Update left position

                            elif v - v_bar < v_star:  # Update v_star if the reduced cost improves
                                v_star = v - v_bar
                                c1_opt, c2_opt = c1, c2 + 1  # Update optimal aperture

                            # Step 3: Increment c2
                            c2 += 1

                        # Return the optimal leaf positions, reduced cost, and beamlets in the aperture
                        optimal_beamlets = row[c1_opt:c2_opt + 1]
                        # left, right, row_score, open_beamlet_indices = self.find_best_leaf_position_fast(row, scores)
                        # left, right, row_score, open_beamlet_indices = self.find_best_leaf_position_fast(row, scores)
                        cumulative_score += v_star
                        selected_beamlets[beam['beam_id']].extend(optimal_beamlets)

                        # remove the best row from configuration
                        if apt_reg_type_cg == 'greedy':

                            selected_leafs[beam['beam_id']][remaining_rows[r_best]] = [c1_opt - 1, c2_opt + 1]  # vmat scp format
                            processed_rows.append(remaining_rows[r_best])
                            del remaining_rows[r_best]
                        else:
                            selected_leafs[beam['beam_id']][r_best] = [c1_opt - 1, c2_opt + 1]
                            processed_rows.append(remaining_rows[0])
                            del remaining_rows[0]

                        # update the score of other rows beamlets based on smoothness penalty with discount factor
                        # Step 2: Penalize remaining rows based on the best row
                        if apt_reg_type_cg == 'greedy':
                            prev_c1_opt = selected_leafs[beam['beam_id']][processed_rows[-1]][0] + 1
                            prev_c2_opt = selected_leafs[beam['beam_id']][processed_rows[-1]][1] - 1
                            for r in remaining_rows:
                                row = beam['reduced_2d_grid'][r]
                                row_distance = abs(r - processed_rows[-1]) - 1  # subtracted one to keep row discount exp(0)=1 for the nearest row
                                row_discount = np.exp(-smooth_beta * row_distance)  # Exponential discount based on row distance
                                # Define the boundary window size. It is half of BEV length of open aperture row

                                for c, beamlet in enumerate(row):
                                    if beamlet >= 0:
                                        # left beamlets
                                        if c < prev_c1_opt:  # left of previous left
                                            score_dict[beamlet] += smooth_delta * row_discount * abs(prev_c1_opt - c)  # decrease score to close the beamlet
                                        elif c > prev_c2_opt:  # right of previous right
                                            score_dict[beamlet] += smooth_delta * row_discount * abs(c - prev_c2_opt)  # decrease score
                                        elif prev_c1_opt < c < prev_c2_opt:  # right of previous left
                                            if (c - prev_c1_opt) < (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1 * smooth_delta * row_discount * abs(prev_c1_opt - c)  # increase score (-ve of distance)
                                            if (c - prev_c1_opt) > (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1 * smooth_delta * row_discount * abs(c - prev_c2_opt)  # increase score (-ve of distance)

                            # beam['scores_matrix'] = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                        else:
                            # Add penalty for deviations from the upper row
                            # Define the boundary window size. It is half of BEV length of open aperture row
                            if r_best < beam['num_rows'] - 1:
                                row = beam['reduced_2d_grid'][r_best + 1]
                                prev_c1_opt = selected_leafs[beam['beam_id']][r_best][0] + 1
                                prev_c2_opt = selected_leafs[beam['beam_id']][r_best][1] - 1
                                for c, beamlet in enumerate(row):
                                    if beamlet >= 0:
                                        # left beamlets
                                        if c < prev_c1_opt:  # left of previous left
                                            score_dict[beamlet] += smooth_delta * abs(prev_c1_opt - c)  # decrease score to close the beamlet
                                        elif c > prev_c2_opt:  # right of previous right
                                            score_dict[beamlet] += smooth_delta * abs(c - prev_c2_opt)  # decrease score
                                        elif prev_c1_opt < c < prev_c2_opt:  # right of previous left
                                            if (c - prev_c1_opt) < (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1 * smooth_delta * abs(prev_c1_opt - c)  # increase score (-ve of distance)
                                            if (c - prev_c1_opt) > (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1 * smooth_delta * abs(c - prev_c2_opt)  # increase score (-ve of distance)
                        aperture_score[beam['beam_id']] = cumulative_score
        # best beam id
        # best_beam_id = min(aperture_score, key=aperture_score.get)
        if 'num_select_apertures' in self.vmat_params:
            num_select_apertures = self.vmat_params['num_select_apertures']
        else:
            num_select_apertures = 1
        top_beam_ids = sorted(aperture_score, key=aperture_score.get)[:num_select_apertures]
        # print('Best reduced cost is of beam id ', best_beam_id, ':', str(aperture_score[best_beam_id]))
        for i, beam_id in enumerate(top_beam_ids, 1):
            print(f'#{i} best reduced cost is of beam ID {beam_id}: {aperture_score[beam_id]}')
        # save leaf positions for the best found beam to arcs dictionary for vmat scp optimization
        for arc in self.arcs.arcs_dict['arcs']:
            for beam in arc['vmat_opt']:
                if beam['beam_id'] in top_beam_ids:
                    beam['leaf_pos_cg'] = selected_leafs[beam['beam_id']]
                    # update leaf positions in arcs dictionary to be used for vmat scp
                    beam['leaf_pos_f'] = selected_leafs[beam['beam_id']]
                    beam['leaf_pos_b'] = selected_leafs[beam['beam_id']]
                    beam['leaf_pos_left'] = [selected_leafs[beam['beam_id']][i][0] for i in range(len(selected_leafs[beam['beam_id']]))]
                    beam['leaf_pos_right'] = [selected_leafs[beam['beam_id']][i][1] for i in range(len(selected_leafs[beam['beam_id']]))]
                    # beam['apt_sol_gen'] = np.isin(beam['reduced_2d_grid'], selected_beamlets[beam['beam_id']]).astype(int)
                    beam['scores_matrix'] = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                    # beam['aperture_score'] = aperture_score[beam['beam_id']]
        return top_beam_ids, [selected_beamlets[b_id] for b_id in top_beam_ids]

    def solve_rmp(self, *args, **kwargs):
        """
        solve restricted master problem to optimize beam intensities
        :param args: solver arguments
        :param kwargs: solver key arguments
        :return: solution. dictionary containing optimal beam intensities and objective value
        """
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        # Create cvxpy problem for RMP
        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        A = inf_matrix.A
        B = self.B
        num_fractions = clinical_criteria.get_num_of_fractions()

        if np.all(self.B == 0):
            return {'y': np.zeros(self.B.shape[1]), 'obj_value': 0}

        st = inf_matrix
        y = cp.Variable(self.B.shape[1], pos=True)
        print('Objective Start')
        self.obj = []
        self.constraints = []
        obj = self.obj
        constraints = self.constraints
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    dose_gy = self.get_dose_gy_per_frac_for_opt(obj_funcs[i], struct=struct)
                    dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dO))]
                    constraints += [B[st.get_opt_voxels_idx(struct), :] @ y <= dose_gy + dO]
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    dose_gy = self.get_dose_gy_per_frac_for_opt(obj_funcs[i], struct=struct)
                    dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dU))]
                    constraints += [B[st.get_opt_voxels_idx(struct), :] @ y >= dose_gy - dU]
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    obj += [self.vmat_params['normalize_oar_weights'] * (1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(B[st.get_opt_voxels_idx(struct), :] @ y))]
        constraint_def = self.constraint_def
        self.constraints_ind_map = {}  # empty previous values if any
        constraints_ind_map = self.constraints_ind_map
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    if org in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(org)) == 0:
                            continue
                        constraints += [B[st.get_opt_voxels_idx(org), :] @ y <= limit / num_fractions]
                        constraints_ind_map['max_dose_' + org + str(limit / num_fractions)] = len(constraints) - 1
            # elif constraint_def[i]['type'] == 'mean_dose':
            #     limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
            #     if limit_key:
            #         limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
            #         org = constraint_def[i]['parameters']['structure_name']
            #         # mean constraints using voxel weights
            #         if org in my_plan.structures.get_structures():
            #             if len(st.get_opt_voxels_idx(org)) == 0:
            #                 continue
            #             fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(org)
            #             limit = limit/fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
            #             constraints += [(1 / sum(st.get_opt_voxels_volume_cc(org))) *
            #                             (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(org),
            #                                                  A[st.get_opt_voxels_idx(org), :] @ x))))
            #                             <= limit / num_fractions]
            elif constraint_def[i]['type'] == 'DFO':
                dfo, oar_voxels = self.get_dfo_parameters(dfo_dict=constraint_def[i], is_obj=False)
                constraints += [B[oar_voxels, :] @ y <= dfo / num_fractions]
                constraints_ind_map['max_dfo'] = len(constraints) - 1
        problem = cp.Problem(cp.Minimize(cp.sum(self.obj)), constraints=self.constraints)

        print('Running Optimization..')
        t = time.time()
        problem.solve(*args, **kwargs)
        sol = {'y': y.value if y.shape[0] != 1 else np.array(y.value), 'obj_value': problem.value}
        elapsed = time.time() - t
        print("Elapsed time: {}".format(elapsed))
        print("Optimal value: %s" % problem.value)

        return sol

    def run_col_gen_algo(self, *args, **kwargs):

        # Initial parameters
        A_indices = []  # Indices of selected beamlets in A
        # preprocessing step to store BEV leaf position
        self.arcs.get_initial_leaf_pos(initial_leaf_pos='BEV')
        # get apt reg metric for BEV
        apt_reg_obj = get_apt_reg_metric(my_plan=self.my_plan)
        print('####apr_reg_obj BEV: {} ####'.format(apt_reg_obj))
        arcs = self.arcs.arcs_dict['arcs']

        self.B = np.empty((self.inf_matrix.A.shape[0], 0))  # Start with an empty B matrix
        A = self.inf_matrix.A

        # get all beam ids from all arcs
        remaining_beam_ids = [beam['beam_id'] for arc in self.arcs.arcs_dict['arcs'] for beam in arc['vmat_opt']]
        map_beam_id_to_index = {
            beam_id: index
            for index, beam_id in enumerate(
                beam_id for arc in arcs for beam_id in arc['beam_ids']
            )
        }
        B_indices = []
        # initialize sol to 0
        sol = {'y': np.zeros(0)}

        obj_funcs = self.opt_params['objective_functions'] if 'objective_functions' in self.opt_params else []
        opt_params_constraints = self.opt_params['constraints'] if 'constraints' in self.opt_params else []
        self.obj_funcs = obj_funcs
        constraint_def = deepcopy(self.clinical_criteria.get_criteria())  # get all constraints definition using clinical criteria

        # add/modify constraints definition if present in opt params
        for opt_constraint in opt_params_constraints:
            # add constraint
            param = opt_constraint['parameters']
            if param['structure_name'] in self.my_plan.structures.get_structures():
                criterion_exist, criterion_ind = self.clinical_criteria.check_criterion_exists(opt_constraint, return_ind=True)
                if criterion_exist:
                    constraint_def[criterion_ind] = opt_constraint
                else:
                    constraint_def += [opt_constraint]
        self.constraint_def = constraint_def

        convergence = []
        # smoothness weight to modify the score. Default is 0
        if 'smooth_delta' in self.vmat_params:
            smooth_delta = self.vmat_params['smooth_delta']
        else:
            smooth_delta = 0
        print('smooth delta: {}'.format(smooth_delta))
        if 'smooth_beta' in self.vmat_params:
            smooth_beta = self.vmat_params['smooth_beta']
        else:
            smooth_beta = 1
        if 'apt_reg_type_cg' in self.vmat_params:
            apt_reg_type_cg = self.vmat_params['apt_reg_type_cg']
        else:
            apt_reg_type_cg = 'row-by-row'
        if 'normalize_oar_weights' not in self.vmat_params:
            self.vmat_params['normalize_oar_weights'] = 1
        # run col generation
        while len(remaining_beam_ids) > 0:

            # Calculate scores for unselected beamlets using gradient information
            scores = self.calculate_scores_grad(A, A_indices, sol)
            # select best aperture
            top_beam_ids, beam_beamlets_list = self.select_best_aperture(remaining_beam_ids, scores, smooth_delta=smooth_delta, smooth_beta=smooth_beta, apt_reg_type_cg=apt_reg_type_cg)

            # create new influence matrix B containing # of cols equals to selected beams
            for beam_id, open_beamlet_indices in zip(top_beam_ids, beam_beamlets_list):
                A_indices.append(open_beamlet_indices)
                B_indices.append(map_beam_id_to_index[beam_id])
                # Sum beamlets for this beam ID and add as new column to B matrix
                self.B = np.column_stack([self.B, A[:, open_beamlet_indices].sum(axis=1).A1])
            # Solve RMP
            sol = self.solve_rmp(*args, **kwargs)  # Use only selected beamlets
            # # save few statistics to convergence list
            convergence.append([top_beam_ids[0],
                                get_apt_reg_metric(self.my_plan, beam_ids=top_beam_ids),
                                sol['obj_value']])
            # Remove used control point
            # remaining_beam_ids.remove(best_beam_id)
            remaining_beam_ids = list(set(remaining_beam_ids) - set(top_beam_ids))

        # calculate beamlet intensity for all the control points as 1d vector
        x = np.zeros(A.shape[1])
        for b in range(len(B_indices)):
            x[A_indices[b]] = sol['y'][b]
        sol['optimal_intensity'] = x

        # Col Gen Convergence
        # df = pd.DataFrame(convergence)
        # print(df)

        # get smoothness function value using leaf positions in arcs
        apt_reg_obj = get_apt_reg_metric(my_plan=self.my_plan)
        print('####apr_reg_obj after col gen: {} ####'.format(apt_reg_obj))
        sol['apt_reg_obj'] = apt_reg_obj

        return sol

    def get_dfo_interior(self, struct_name: str = 'GTV', min_dose: float = None, max_dose: float = None, pres: float = None):

        # get boundary and calc distance for interior voxels
        voxels = self.inf_matrix.get_opt_voxels_idx(struct_name)
        if min_dose is not None and max_dose is not None:
            if 'dfo_target_interior' not in self.clinical_criteria.clinical_criteria_dict:
                self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'] = {}
            if struct_name not in self.clinical_criteria.clinical_criteria_dict['dfo_target_interior']:
                # Assuming `target_mask` is your 3D binary mask with 1s inside all target structures
                target_mask = self.my_plan.structures.get_structure_mask_3d(struct_name)

                # Step 1: Label each sub-region in the mask
                labeled_mask, num_regions = label(target_mask)

                # Set up parameters for distance computation
                voxel_resolution = np.array(self.inf_matrix.opt_voxels_dict['ct_voxel_resolution_xyz_mm'][::-1])
                ct_origin = np.array(self.inf_matrix.opt_voxels_dict['ct_origin_xyz_mm'][::-1])

                # Get all GTV voxel coordinates in physical space
                vox_coord_xyz_mm = self.inf_matrix.opt_voxels_dict['voxel_coordinate_XYZ_mm'][0]
                interior_points = vox_coord_xyz_mm[voxels, :]  # All GTV voxel coordinates

                # List to accumulate boundary coordinates from each region
                all_boundary_coords = []

                for region_id in range(1, num_regions + 1):
                    # Extract the mask for the current region
                    region_mask = (labeled_mask == region_id)

                    # Identify boundary voxels for this region
                    eroded_region = binary_erosion(region_mask)
                    boundary_mask = region_mask & ~eroded_region
                    boundary_voxels = np.argwhere(boundary_mask)

                    # Convert boundary voxels to physical coordinates
                    boundary_coords = boundary_voxels * voxel_resolution + ct_origin
                    boundary_coords = boundary_coords[:, [2, 1, 0]]  # Convert ZYX to XYZ

                    # Accumulate boundary coordinates for this region
                    all_boundary_coords.append(boundary_coords)

                # Combine all boundary coordinates into a single array
                all_boundary_coords = np.vstack(all_boundary_coords)

                # Step 4: Use Nearest Neighbors to find the distance from each interior point to the nearest boundary point
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(all_boundary_coords)
                distances_voxels, _ = nbrs.kneighbors(interior_points)

                def fit_exponential_growth(x1, y1, x2, y2):
                    b = (np.log(y2) - np.log(y1)) / (x2 - x1)
                    a = y1 / np.exp(b * x1)
                    return a, b

                # Calculate a and b based on the given points
                a, b = fit_exponential_growth(np.min(distances_voxels), min_dose, np.max(distances_voxels), max_dose)

                prescription = np.squeeze(a * np.exp(b * distances_voxels))
                self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'][struct_name] = prescription
                # self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'] = {struct_name + 'distance_from_boundary_mm': distances_voxels}
            else:
                prescription = self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'][struct_name]
        else:
            prescription = np.repeat(pres, len(voxels))
        return prescription

    def get_dose_gy_per_frac_for_opt(self, obj, struct):
        num_fractions = self.my_plan.get_num_of_fractions()
        dose_key = self.matching_keys(obj, 'dose_')
        dosemin_key = self.matching_keys(obj, 'dosemin')
        dosemax_key = self.matching_keys(obj, 'dosemax')
        dose_gy = None
        if dose_key:
            dose_gy = self.dose_to_gy(dose_key, obj[dose_key]) / num_fractions
        elif dosemax_key:
            dosemin_gy = self.dose_to_gy(dosemin_key, obj[dosemin_key]) / num_fractions
            dosemax_gy = self.dose_to_gy(dosemax_key, obj[dosemax_key]) / num_fractions
            dose_gy = self.get_dfo_interior(struct_name=struct, min_dose=dosemin_gy, max_dose=dosemax_gy)
        return dose_gy

    def get_dfo_parameters(self, dfo_dict, is_obj: bool = False):
        weight_interpolate = None
        if not is_obj:
            param = dfo_dict['parameters']
            struct_name = param['structure_name']
            key = self.matching_keys(dfo_dict['constraints'], 'dose')
            max_dose = np.asarray([self.dose_to_gy(key, dose) for dose in dfo_dict['constraints'][key]])
            distance = np.asarray(param['distance_from_structure_mm'])
        else:
            struct_name = dfo_dict['structure_name']
            distance = np.asarray(dfo_dict['distance_from_structure_mm'])
            key = self.matching_keys(dfo_dict, 'dose')
            max_dose = np.asarray([self.dose_to_gy(key, dose) for dose in dfo_dict[key]])
            weight = np.asarray(dfo_dict['weight'])
            weight_interpolate = interp1d(distance, weight, kind='next')
        dfo_interpolate = interp1d(distance, max_dose, kind='next')
        target_voxels = self.inf_matrix.get_opt_voxels_idx(struct_name)
        all_vox = np.arange(self.inf_matrix.A.shape[0])
        oar_voxels = all_vox[~np.isin(np.arange(self.inf_matrix.A.shape[0]), target_voxels)]
        vox_coord_xyz_mm = self.inf_matrix.opt_voxels_dict['voxel_coordinate_XYZ_mm'][0]

        if 'distance_from_structure_mm' not in self.inf_matrix.opt_voxels_dict:
            self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'] = {}
        if struct_name not in self.inf_matrix.opt_voxels_dict['distance_from_structure_mm']:
            print('calculating distance of normal tissue voxels from target for DFO constraints. This step may take some time..')
            start = time.time()
            a, _ = cKDTree(vox_coord_xyz_mm[target_voxels, :]).query(vox_coord_xyz_mm[oar_voxels, :], 1)
            # a = spatial.distance.cdist(, vox_coord_xyz_mm[PTV, :]).min(axis=1)
            print('Time for calc distance {}'.format(time.time() - start))
            dist_from_structure = np.zeros_like(all_vox, dtype=float)
            dist_from_structure[oar_voxels] = a
            self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'] = {struct_name: dist_from_structure}
        if not is_obj:
            return dfo_interpolate(self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name][oar_voxels]), oar_voxels
        else:
            return dfo_interpolate(self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name][oar_voxels]), weight_interpolate(
                self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name][oar_voxels]), oar_voxels

    def remove_duplicates(self, dict_list):
        seen = set()
        unique_dicts = []

        for d in dict_list:
            key_str = json.dumps(d, sort_keys=True)  # Convert dict to JSON string (sorted)
            if key_str not in seen:
                seen.add(key_str)
                unique_dicts.append(d)

        return unique_dicts
