import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set

from dataReader import *
from graphSimilarity import *

class ContextAwareRecommendation:
    def __init__(self, source_dir: str, sub_folder: str, num_of_neighbors: int, testing_start_pos: int, testing_end_pos: int):
        self.src_dir = source_dir
        self.sub_folder = sub_folder
        self.num_of_neighbors = num_of_neighbors
        self.ground_truth = os.path.join(self.src_dir, self.sub_folder, "GroundTruth")
        self.rec_dir = os.path.join(self.src_dir, self.sub_folder, "Recommendations")
        self.sim_dir = os.path.join(self.src_dir, self.sub_folder, "Similarities")
        self.testing_start_pos = testing_start_pos
        self.testing_end_pos = testing_end_pos
        self.reader = DataReader()

        self.num_of_slices = self.num_of_rows = self.num_of_cols = None

    def build_user_item_context_matrix(self, testing_pro: str, list_of_projects: List[str], list_of_method_invocations: List[str]) -> np.ndarray:
        sim_projects = self.reader.get_most_similar_projects(os.path.join(self.sim_dir, testing_pro), self.num_of_neighbors)
        # print("SP", sim_projects)

        # print(testing_pro, list_of_projects, list_of_method_invocations)
        
        all_projects = {}
        list_of_prs = []
        all_mds = set()
        all_mis = set()

        for project in sim_projects:
            tmp_mis = self.reader.get_project_details_from_arff2(self.src_dir, sim_projects[project])
            all_mds.update(tmp_mis.keys())
            for mis in tmp_mis.values():
                all_mis.update(mis)
            all_projects[sim_projects[project]] = tmp_mis
            list_of_prs.append(sim_projects[project])

        list_of_prs.append(testing_pro)

        # print(all_mds, all_mis, list_of_prs)

        # print(self.ground_truth, testing_pro)

        ground_truth_mis = self.reader.get_ground_truth_invocations(self.ground_truth, testing_pro)

        testing_mis = {}
        tmp_mis = self.reader.get_testing_project_details(self.src_dir, testing_pro, ground_truth_mis, testing_mis)

        # print(testing_mis)

        # print("tmp_mis", tmp_mis)

        all_mds.update(tmp_mis.keys())
        for s in tmp_mis.values():
            all_mis.update(s)

        testing_md = list(testing_mis.keys())[0]
        tmp_mi_set = testing_mis[testing_md]

        tmp_mis.update(testing_mis)
        all_projects[testing_pro] = tmp_mis

        list_of_mds = sorted(all_mds)
        if testing_md in list_of_mds:
            list_of_mds.remove(testing_md)
        list_of_mds.append(testing_md)

        list_of_mis = sorted(all_mis)
        for testing_mi in tmp_mi_set:
            if testing_mi in list_of_mis:
                list_of_mis.remove(testing_mi)
        for testing_mi in tmp_mi_set:
            list_of_mis.append(testing_mi)
        
        self.num_of_slices = len(list_of_prs)
        self.num_of_rows = len(list_of_mds)
        self.num_of_cols = len(list_of_mis)
        matrix = np.zeros((self.num_of_slices, self.num_of_rows, self.num_of_cols), dtype=np.byte)

        # print("2", list_of_mds, list_of_mis, all_projects)

        # Populate the matrix with 1s where applicable
        for i in range(self.num_of_slices - 1):
            current_pro = list_of_prs[i]
            myMDs = all_projects.get(current_pro, {})

            for j in range(self.num_of_rows):
                current_MD = list_of_mds[j]

                if current_MD in myMDs:
                    myMIs = myMDs[current_MD]

                    for k in range(self.num_of_cols):
                        current_MI = list_of_mis[k]

                        if current_MI in myMIs:
                            matrix[i, j, k] = 1

        # Handle the testing project
        myMDs = all_projects.get(testing_pro, {})
        for j in range(self.num_of_rows - 1):
            current_MD = list_of_mds[j]

            if current_MD in myMDs:
                myMIs = myMDs[current_MD]

                for k in range(self.num_of_cols):
                    current_MI = list_of_mis[k]

                    if current_MI in myMIs:
                        matrix[self.num_of_slices - 1, j, k] = 1

        # Final row in the last slice
        current_MD = list_of_mds[-1]
        myMIs = myMDs.get(current_MD, set())
        for k in range(self.num_of_cols):
            current_MI = list_of_mis[k]

            if current_MI in myMIs:
                matrix[self.num_of_slices - 1, self.num_of_rows - 1, k] = 1
            else:
                matrix[self.num_of_slices - 1, self.num_of_rows - 1, k] = -1

        # Adding projects and method invocations to the respective lists
        list_of_projects.extend(list_of_prs)
        list_of_method_invocations.extend(list_of_mis)
        return matrix

    def recommendation(self):
        testing_projects = self.reader.read_project_list(os.path.join(self.src_dir, "List.txt"), self.testing_start_pos, self.testing_end_pos)

        for testing_pro in testing_projects:
            # print(testing_projects[testing_pro])
            recommendations = {}
            list_of_prs = []
            list_of_mis = []

            sim_scores = self.reader.get_similarity_scores(os.path.join(self.sim_dir, testing_projects[testing_pro]), self.num_of_neighbors)
            matrix = self.build_user_item_context_matrix(testing_projects[testing_pro], list_of_prs, list_of_mis)

            # print(sim_scores, matrix)
            # testing_method_vector = matrix[-1, -1]
            testing_method_vector = matrix[self.num_of_slices - 1][self.num_of_rows - 1]
            # print(self.num_of_slices, self.num_of_rows)
            md_sim_scores = {}

            for i in range(self.num_of_slices - 1):
                for j in range(self.num_of_rows):
                    other_method_vector = matrix[i][j]

                    sim_calculator = GraphBasedSimilarityCalculator(self.src_dir)
                    sim = sim_calculator.compute_jaccard_similarity(testing_method_vector, other_method_vector)
                    # print(sim)
                    key = f"{i}#{j}"
                    md_sim_scores[key] = sim

            sim_sorted_map = sorted(md_sim_scores.items(), key=lambda item: item[1], reverse=True)
            top3_sim = dict(sim_sorted_map[:3])

            try:
                ratings = np.zeros(self.num_of_cols - 1)
                for k in range(self.num_of_cols):
                    if matrix[self.num_of_slices - 1][self.num_of_rows - 1][k] == -1:
                        total_sim = 0

                        for key, method_sim in top3_sim.items():
                            slice_idx, row_idx = map(int, key.split("#"))
                            avg_md_rating = np.mean(matrix[slice_idx][row_idx])
                            project = list_of_prs[slice_idx]
                            project_sim = sim_scores[project]
                            val = project_sim * matrix[slice_idx][row_idx][k]

                            total_sim += method_sim
                            try:
                                ratings[k] += (val - avg_md_rating) * method_sim
                            except Exception as e:
                                print(f"Error processing {testing_projects[testing_pro]}: {e}")

                        if total_sim != 0:
                            ratings[k] /= total_sim

                        active_md_rating = 0.8
                        ratings[k] += active_md_rating
                        method_invocation = list_of_mis[k]
                        recommendations[method_invocation] = ratings[k]

            except Exception as e:
                print(f"Error processing {testing_projects[testing_pro]}: {e}")

            rec_sorted_map = dict(sorted(recommendations.items(), key=lambda item: item[1], reverse=True))

            if not os.path.exists(self.rec_dir):
                os.makedirs(self.rec_dir)

            self.reader.write_recommendations(os.path.join(self.rec_dir, testing_projects[testing_pro]), rec_sorted_map, recommendations)
            