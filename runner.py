import os
import time
import logging
import sys
from collections import defaultdict

from similarity import *
from graphSimilarity import *
from similarityCalculator import *
from cars import *
from successCalculator import *

from configuration import *

class Runner:
    def __init__(self):
        self.src_dir = None
        self.num_of_projects = 0
        self.ten_fold = False
        self.leave_one_out = False
        self.configuration = None
        self.pam = False

    def load_configurations(self, prop_file):
        """
        Loads configurations from a properties file and sets the relevant attributes for the class.
        
        :param prop_file: Path to the properties file containing configuration settings.
        :return: True if the configurations were loaded successfully, False otherwise.
        
        This method performs the following actions:
        
        - Reads the properties file and parses it into a dictionary.
        - Sets the source directory using the 'sourceDirectory' key from the properties file.
        - Validates and sets the configuration type based on predefined configurations.
        - Validates and sets the validation mode, either 'ten-fold' or 'leave-one-out'.
        - Counts the number of projects by reading the 'List.txt' file in the source directory.
        
        If the file cannot be read, the method logs an error and returns False."""

        try:
            with open(prop_file, 'r') as f:
                # Parse the properties file into a dictionary
                prop = dict(line.strip().split(':') for line in f if ':' in line)
            
            print(prop)
            # Set source directory
            self.src_dir = prop.get('sourceDirectory')

            conf = prop.get("configuration")
            if conf == "C1.1":
                self.configuration = Configuration.C1_1
            elif conf == "C1.2":
                self.configuration = Configuration.C1_2
            elif conf == "C2.1":
                self.configuration = Configuration.C2_1
            elif conf == "C2.2":
                self.configuration = Configuration.C2_2
            else:
                log.error(f"Invalid configuration {conf}")


            # Set the validation mode
            mode = prop.get('validation')
            if mode == "ten-fold":
                self.ten_fold = True
            
            elif mode == "leave-one-out":
                self.leave_one_out = True
            
            else:
                logging.error(f"Invalid validation mode {mode}")

            # Count the number of projects by reading the project list
            project_list_path = os.path.join(self.src_dir, 'List.txt')
            with open(project_list_path, 'r') as reader:
                self.num_of_projects = sum(1 for _ in reader)

            return True
        except IOError as e:
            logging.error("Couldn't read evaluation.properties", exc_info=e)
            return False
    
    def run(self, args):
        
        logging.info("FOCUS: A Context-Aware Recommender System!");

        # Default properties file
        prop_file = "./properties.yaml"

        # Check command-line arguments
        if len(args) == 1:
            if args[0].upper() == "PAM":
                self.pam = True
            else:
                prop_file = args[0]

        # If PAM mode is enabled, perform PAM calculations
        if self.pam:
            self.src_dir = "../../dataset/PAM/SH_S-results/"
            logging.info(f"Computing PAM results from {self.src_dir} (leave-one-out cross-validation)")
            self.calculate_success_pam()
            return

        # Load configurations from the properties file
        if self.load_configurations(prop_file):
            if self.ten_fold:
                ks = [1, 5, 10, 15, 20]
                for k in ks:
                    logging.info(f"Running the evaluation with k = {k}")
                    before = time.time()
                    self.ten_fold_cross_validation(k, "Structural")
                    after = time.time()
                    logging.info(f"Evaluation with k = {k} took {after - before:.2f} seconds")
                    # exit()

            if self.leave_one_out:
                before = time.time()
                logging.info(f"Running leave-one-out cross-validation on {self.src_dir} with configuration {self.configuration}")
                self.leave_one_out_validation()
                after = time.time()
                logging.info(f"Leave-one-out took {after - before:.2f} seconds")
        else:
            logging.error("Aborting due to configuration loading failure.")

    def ten_fold_cross_validation(self, num_of_neighbors, similarity_type):
        """
        Perform a ten-fold cross-validation process.

        :param num_of_neighbors: Number of neighbors to consider for the recommendation engine.
        :param similarity_type: Similarity metric to be used.
        """
        logging.info("Starting 10-fold cross-validation...")
        num_of_folds = 10
        step = self.num_of_projects // 10
        # print(step, self.num_of_projects)
        ns = list(range(1, 21))
        avg_success = defaultdict(float)
        avg_precision = defaultdict(float)
        avg_recall = defaultdict(float)

        for i in range(num_of_folds):
            start_time = time.time()

            # print(i, step)

            training_start_pos1 = 1
            training_end_pos1 = i * step
            training_start_pos2 = (i + 1) * step + 1
            training_end_pos2 = self.num_of_projects
            testing_start_pos = 1 + i * step
            testing_end_pos = (i + 1) * step
            # print(training_start_pos1, training_end_pos1, training_start_pos2, training_end_pos2, testing_start_pos, testing_end_pos)
            k = i + 1
            sub_folder = f"evaluation/round{k}"

            # print(similarity_type,
            # self.src_dir, sub_folder, 
            # self.configuration, training_start_pos1,
            # training_end_pos1, training_start_pos2,
            # training_end_pos2, testing_start_pos, testing_end_pos)

            # exit()

            # training_start_pos1 = 1
            # training_end_pos1 = 1712
            # training_start_pos2 = 1927
            # training_end_pos2 = 2148
            # testing_start_pos = 1713
            # testing_end_pos = 1926

            # print(similarity_type,
            # self.src_dir, sub_folder, 
            # self.configuration, training_start_pos1,
            # training_end_pos1, training_start_pos2,
            # training_end_pos2, testing_start_pos, testing_end_pos)

            # Depending on the similarity type, initialize the similarity calculator
            if similarity_type == Similarity.SYNTACTICALLY:
                calculator = GraphBasedSimilarityCalculator(self.src_dir, sub_folder,
                                                            self.configuration, training_start_pos1, training_end_pos1, 
                                                            training_start_pos2, training_end_pos2,
                                                            testing_start_pos, testing_end_pos)
            else:
                calculator = GraphBasedSimilarityCalculator(self.src_dir, sub_folder,
                                                            self.configuration, training_start_pos1, training_end_pos1, 
                                                            training_start_pos2, training_end_pos2,
                                                            testing_start_pos, testing_end_pos)
            calculator.compute_project_similarity()

            engine = ContextAwareRecommendation(self.src_dir, sub_folder, num_of_neighbors,
                                                testing_start_pos, testing_end_pos)
            engine.recommendation()
            elapsed_time = time.time() - start_time
            logging.info("\tFold %d time %.2f ms", i, elapsed_time * 1000)

            calc = SuccessCalculator(self.src_dir, sub_folder, testing_start_pos, testing_end_pos)
            for n in ns:
                success = calc.compute_success_rate(n)
                precision = calc.compute_precision(n)
                recall = calc.compute_recall(n)
                avg_success[n] += success
                avg_precision[n] += precision
                avg_recall[n] += recall

                # print(success, precision, recall)

        logging.info("### 10-FOLDS RESULTS ###")
        logging.info("N, SR, P, R, Neighbors")
        for n in ns:
            logging.info("%d\t%.3f\t%.3f\t%.3f\t%d", 
                        n, 
                        avg_success[n] / num_of_folds, 
                        avg_precision[n] / num_of_folds, 
                        avg_recall[n] / num_of_folds, 
                        num_of_neighbors)
        
        # exit()


if __name__ == "__main__":
    runner = Runner()
    runner.run(sys.argv[1:])



# Set up logging to print to console
logging.basicConfig(level=logging.INFO)

# Initialize the Runner class
runner = Runner()

# # Test the load_configurations method
success = runner.run([])