import os
from collections import defaultdict, OrderedDict
from typing import Dict, Set

import logging
# Configure the logger
logging.basicConfig(level=logging.DEBUG,  # Set the minimum level of severity to DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataReader:

    def __init__(self):
        self.log = logging.getLogger("DataReader_Class")

    def read_project_list(self, filename, start_pos, end_pos):
        """
        Reads a list of projects from a file, returning a dictionary mapping an index to each project name.
        :param filename: Path to the file containing the list of projects.
        :param start_pos: The starting position to read from.
        :param end_pos: The ending position to read up to.
        :return: A dictionary where the key is an index and the value is a project name.
        """
        ret = {}
        count = 1
        project_id = start_pos

        try:
            with open(filename, 'r') as reader:
                # Skip lines until we reach the start position
                while start_pos != -1 and count < start_pos:
                    line = reader.readline()
                    count += 1
                
                # Read lines from start_pos to end_pos (or the end of the file)
                line = reader.readline()
                while line != '':
                    repo = line.strip()
                    ret[project_id] = repo
                    project_id += 1
                    count += 1
                    if end_pos != -1 and count > end_pos:
                        break
                    
                    line = reader.readline()
        
        except IOError as e:
            # print(f"Couldn't read file {filename}: {e}")
            self.log.error(f"Couldn't read file {filename}: {e}", exc_info=True)

        return ret

    
    def read_recommendation_file(self, filename, size):
        """
        Reads a specific number of recommendations from a file and returns them as a set.
        :param filename: The path to the file containing the recommendations.
        :param size: The number of recommendations to read from the file.
        :return: A set containing the first `size` recommendations from the file.
        """
        recommendations = set()
        count = 0

        try:
            with open(filename, 'r') as file:
                for line in file:
                    vals = line.split("\t")
                    library = vals[0].strip()
                    recommendations.add(library)
                    count += 1
                    if count == size:
                        break
        except IOError as e:
            # print(f"Couldn't read file {filename}: {e}")
            self.log.error(f"Couldn't read file {filename}: {e}", exc_info=True)

        return recommendations


    
    def read_ground_truth_invocations(self, filename):
        """
        Reads the ground truth invocations from a file, returning a set of invocations.
        :param filename: Path to the ground truth file.
        :return: A set of ground truth invocations.
        """
        ret = set()

        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("#")
                    invocation = vals[1].strip()
                    ret.add(invocation)
        except IOError as e:
            # print(f"Couldn't read file {filename}: {e}")
            self.log.error(f"Couldn't read file {filename}: {e}", exc_info=True)

        return ret
    
    def get_project_invocations(self, path, name):
        """
        Reads a file containing method invocations and returns a dictionary 
        mapping the project name to another dictionary of method invocation frequencies.

        :param path: The directory path where the file is located.
        :param name: The name of the file containing the method invocations.
        :return: A dictionary mapping the project name to its method invocations and their frequencies.
        """
        method_invocations = defaultdict(dict)
        terms = defaultdict(int)
        filename = os.path.join(path, name)

        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    parts = line.strip().split("#")
                    mi = parts[1].strip()

                    # Update frequency of the method invocation
                    terms[mi] += 1

        except Exception as e:
            # self.log.error(f"Couldn't read file {filename}", exc_info=True)
            self.log.error(f"Couldn't read file {filename}: {e}", exc_info=True)

        method_invocations[name] = dict(terms)
        return method_invocations

    def get_project_details2(self, path, filename):
        method_invocations = OrderedDict()  # Equivalent of LinkedHashMap in Java
        filename = os.path.join(path, filename)
        prev_md = ""
        current_md = ""
        start = True
        done_declarations = set()

        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    parts = line.strip().split("#")
                    md = parts[0].strip()
                    mi = parts[1].strip()

                    # To avoid a project with two identical method declarations
                    if start:
                        prev_md = md
                        start = False
                    current_md = md

                    if current_md != prev_md:
                        done_declarations.add(prev_md)
                        prev_md = current_md

                    if current_md not in done_declarations:
                        if md in method_invocations:
                            vector = method_invocations[md]
                        else:
                            vector = []
                        vector.append(mi)
                        method_invocations[md] = vector
        except Exception as e:
            log.error(f"Couldn't read file {filename}", exc_info=True)

        # Remove all declarations with fewer than 2 invocations from the data
        temp = [key for key in method_invocations if len(method_invocations[key]) < 2]

        for key in temp:
            method_invocations.pop(key)
        
        # print(method_invocations)
        return method_invocations
    
    def get_testing_project_invocations(self, path, sub_folder, filename, num_of_invocations, remove_half):
        # print(num_of_invocations)
        method_invocations = self.get_project_details2(path, filename)

        key_set = set(method_invocations.keys())
        removed_key = set()

        # Remove the last half of the method declarations if there are more than 5
        if len(key_set) < 3:
            remove_half = False

        if remove_half:
            size = len(method_invocations)
            half = round(size / 2)
            count = 0
            for key in key_set:
                count += 1
                if count > half:
                    removed_key.add(key)

        # Remove the last declarations
        for key in removed_key:
            method_invocations.pop(key, None)

        key_set = set(method_invocations.keys())
        key_list = list(key_set)
        size = len(key_list)

        # Select the last method as testing
        index = size - 1
        testing_declaration = key_list[index]
        ground_truth_mis = set()
        invocation_list = method_invocations[testing_declaration]
        size = len(invocation_list)

        invocation_set = set(invocation_list)

        # print("invlist:", invocation_list)

        query = invocation_list[:num_of_invocations]
        # print(query, num_of_invocations, size)
        ground_truth_mis.update(invocation_list[num_of_invocations:])
        # print(ground_truth_mis, "hds")

        # Remove the testing declaration from the set of method invocations
        method_invocations.pop(testing_declaration, None)

        # Save back the testing declaration with the selected invocations
        tmp_method_invocations = {testing_declaration: invocation_list}
        method_invocations.update(tmp_method_invocations)

        testing_invocation_location = os.path.join(path, sub_folder, "TestingInvocations")
        os.makedirs(testing_invocation_location, exist_ok=True)

        # Save the testing method invocations to an external file for future usage
        try:
            with open(os.path.join(testing_invocation_location, filename), 'w') as writer:
                for invocation in query:
                    content = f"{testing_declaration}#{invocation}"
                    writer.write(content + '\n')
        except IOError as e:
            print(f"Couldn't read file {testing_invocation_location}{filename}: {e}")

        ground_truth_path = os.path.join(path, sub_folder, "GroundTruth")
        os.makedirs(ground_truth_path, exist_ok=True)

        # Save the ground-truth method invocations to an external file for future comparison
        # print("gtm", ground_truth_mis)
        # exit()
        try:
            with open(os.path.join(ground_truth_path, filename), 'w') as writer:
                for s in ground_truth_mis:
                    content = f"{testing_declaration}#{s}"
                    writer.write(content + '\n')
        except IOError as e:
            print(f"Couldn't read file {ground_truth_path}{filename}: {e}")

        ret = {}
        map = defaultdict(int)

        # Get all method invocations and their corresponding frequency
        for key, terms in method_invocations.items():
            for term in terms:
                map[term] += 1

        ret[filename] = dict(map)
        return ret
    
    def write_similarity_scores(self, sim_dir, project, similarities):
        filename = os.path.join(sim_dir, project)

        try:
            with open(filename, 'w') as writer:
                for project_name, similarity_score in similarities.items():
                    writer.write(f"{project}\t{project_name}\t{similarity_score}\n")
                    writer.flush()
        except IOError as e:
            log.error(f"Couldn't write file {filename}", exc_info=True)
    


    def get_project_details_from_arff2(self, path: str, filename: str) -> Dict[str, Set[str]]:
        method_invocations = {}
        filename = os.path.join(path, filename)
        count = 0

        try:
            with open(filename, 'r') as file:
                for line in file:
                    count += 1
                    if count > 6:
                        parts = line.split('#')
                        md = parts[0].replace("'", "").strip()
                        temp = parts[1].replace("'", "").strip()

                        invocations = temp.split()
                        vector = method_invocations.get(md, set())

                        for mi in invocations:
                            mi = mi.strip()
                            if mi:
                                vector.add(mi)

                        method_invocations[md] = vector

        except IOError as e:
            print(f"Couldn't read file {filename}: {e}")

        return method_invocations
    
    def get_most_similar_projects(self, filename: str, size: int) -> Dict[int, str]:
        projects = {}
        count = 0

        try:
            with open(filename, 'r') as file:
                for line in file:
                    vals = line.split('\t')
                    if len(vals) > 1:
                        uri = vals[1].strip()
                        projects[count] = uri
                        count += 1
                        if count == size:
                            break
        except IOError as e:
            print(f"Couldn't read file {filename}: {e}")

        return projects

    def get_similarity_scores(self, filename: str, size: int) -> Dict[str, float]:
        projects = {}
        count = 0
        try:
            with open(filename, 'r') as file:
                for line in file:
                    vals = line.split('\t')
                    if len(vals) > 2:
                        uri = vals[1].strip()
                        try:
                            score = float(vals[2].strip())
                            projects[uri] = score
                            count += 1
                            if count == size:
                                break
                        except ValueError:
                            print(f"Error parsing float value in line: {line}")
        except IOError as e:
            print(f"Couldn't read file {filename}: {e}")

        return projects
    
    def get_ground_truth_invocations(self, path: str, filename: str) -> Set[str]:
        gt_invocations = set()
        file_path = os.path.join(path, filename)

        # print("file", file_path)
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # print(line)
                    gt_invocations.add(line.strip())
        except IOError as e:
            print(f"Couldn't read file {file_path}: {e}")
        
        # print("GTI", gt_invocations)
        return gt_invocations
    
    def get_testing_project_details(self, path: str, filename: str, gt_invocations: Set[str],
                                 testing_mis: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        method_invocations = {}
        testing_md = ""
        testing_mi = ""

        # Get the testing method declaration
        for s in gt_invocations:
            parts = s.split("#")
            testing_md = parts[0].strip()
            break

        # print(testing_md)

        tmp = set()
        file_path = path + filename

        # print("fp", file_path)

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if testing_md in line:
                        # Get the testing method invocations
                        if line not in gt_invocations:
                            parts = line.split("#")
                            testing_mi = parts[1].strip()
                            tmp.add(testing_mi)
                    else:
                        # Other method invocations
                        parts = line.split("#")
                        md = parts[0].strip()
                        mi = parts[1].strip()
                        if md not in method_invocations:
                            method_invocations[md] = set()
                        method_invocations[md].add(mi)
        except IOError as e:
            print(f"Couldn't read file {file_path}: {e}")

        testing_mis[testing_md] = tmp
        # print(testing_mis)
        return method_invocations
    
    def write_recommendations(self, filename: str, sorted_map: dict, recommendations: dict) -> None:
        try:
            with open(filename, 'w') as file:
                for key in sorted_map.keys():
                    file.write(f"{key}\t{recommendations.get(key)}\n")
        except IOError as e:
            print(f"Couldn't write file {filename}: {e}")
