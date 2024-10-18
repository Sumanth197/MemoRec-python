import os
from abc import ABC, abstractmethod
from typing import Dict, Any

from dataReader import *
from configuration import *

class SimilarityCalculator(ABC):
    """
    Abstract base class to calculate similarity between projects.

    :param src_dir: Source directory where the project data is located.
    :param sub_folder: Sub-folder under the source directory for storing results.
    :param conf: Configuration object to determine the processing strategy.
    :param training_start_pos1: Start position for the first set of training data.
    :param training_end_pos1: End position for the first set of training data.
    :param training_start_pos2: Start position for the second set of training data.
    :param training_end_pos2: End position for the second set of training data.
    :param testing_start_pos: Start position for the testing data.
    :param testing_end_pos: End position for the testing data.
    """

    def __init__(self, src_dir: str, sub_folder: str = None, conf: Any = None, 
                 training_start_pos1: int = None, training_end_pos1: int = None, 
                 training_start_pos2: int = None, training_end_pos2: int = None, 
                 testing_start_pos: int = None, testing_end_pos: int = None):
        
        self.src_dir = src_dir
        self.sub_folder = sub_folder
        self.configuration = conf
        self.sim_dir = None
        self.training_start_pos1 = training_start_pos1
        self.training_end_pos1 = training_end_pos1
        self.training_start_pos2 = training_start_pos2
        self.training_end_pos2 = training_end_pos2
        self.testing_start_pos = testing_start_pos
        self.testing_end_pos = testing_end_pos
        self.reader = DataReader()
        if self.sub_folder:
            self.set_sim_dir(os.path.join(self.src_dir, self.sub_folder, "Similarities"))

    @abstractmethod
    def compute_similarity(self, testing_pro: str, projects: Dict[str, Dict[str, int]]):
        """
        Abstract method to compute the similarity between the testing project and the list of training projects.

        :param testing_pro: The ID of the testing project.
        :param projects: A dictionary containing project IDs and their associated data.
        """
        pass

    def compute_project_similarity(self):
        """
        Compute the similarity between all testing projects and training projects.

        This method reads the training and testing projects, applies the configuration settings, 
        and computes the similarity between them.
        """
        # print("entered")
        training_projects = {}
        training_projects_id = {}

        # Read all training project IDs
        if self.training_start_pos1 < self.training_end_pos1:
            training_projects_id.update(self.reader.read_project_list(
                os.path.join(self.src_dir, "List.txt"), 
                self.training_start_pos1, 
                self.training_end_pos1
            ))

        if self.training_start_pos2 < self.training_end_pos2:
            training_projects_id.update(self.reader.read_project_list(
                os.path.join(self.src_dir, "List.txt"), 
                self.training_start_pos2, 
                self.training_end_pos2
            ))

        # Read all training projects
        for training_id in training_projects_id.values():
            training_projects.update(self.reader.get_project_invocations(self.src_dir, training_id))

        # print(training_projects)
        # exit()

        # Read all testing project IDs
        testing_projects_id = self.reader.read_project_list(
            os.path.join(self.src_dir, "List.txt"), 
            self.testing_start_pos, 
            self.testing_end_pos
        )

        num_of_testing_invocations = 0
        remove_half = False

        if self.configuration == Configuration.C1_1:
            num_of_testing_invocations = 1
            remove_half = True
        elif self.configuration == Configuration.C1_2:
            num_of_testing_invocations = 4
            remove_half = True
        elif self.configuration == Configuration.C2_1:
            num_of_testing_invocations = 1
            remove_half = False
        elif self.configuration == Configuration.C2_2:
            num_of_testing_invocations = 4
            remove_half = False
        
        # print(self.configuration, Configuration.C2_1)
        # print(num_of_testing_invocations)

        for testing_id in testing_projects_id.values():
            # Get half of all declarations and use for similarity computation
            testing_project = self.reader.get_testing_project_invocations(
                self.src_dir, self.sub_folder, testing_id, num_of_testing_invocations, remove_half
            )

            training_projects.update(testing_project)
            self.compute_similarity(testing_id, training_projects)
            training_projects.pop(testing_id)

    def get_sim_dir(self) -> str:
        """
        Get the directory path where the similarity results are stored.

        :return: The similarity directory path.
        """
        return self.sim_dir

    def set_sim_dir(self, sim_dir: str):
        """
        Set the directory path for storing similarity results and ensure the directory exists.

        :param sim_dir: The similarity directory path.
        """
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        self.sim_dir = sim_dir