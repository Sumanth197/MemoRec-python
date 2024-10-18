import math
from collections import defaultdict
from logging import getLogger, Formatter, FileHandler

from similarityCalculator import *

class GraphBasedSimilarityCalculator(SimilarityCalculator):
    
    log = getLogger("GraphBasedSimilarityCalculator")
    handler = FileHandler('similarity_calculator.log')
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)
    log.setLevel("DEBUG")
    
    def __init__(self, src_dir, sub_folder=None, conf=None, training_start_pos1=None, training_end_pos1=None,
                 training_start_pos2=None, training_end_pos2=None, testing_start_pos=None, testing_end_pos=None):
        super().__init__(src_dir, sub_folder, conf, training_start_pos1, training_end_pos1, training_start_pos2,
                         training_end_pos2, testing_start_pos, testing_end_pos)

    def compute_similarity(self, testing_pro, projects):
        """
        Compute the similarity between the testing project and all other training projects.
        The results are serialized for further analysis.

        :param testing_pro: The project that needs to be tested against all other projects.
        :param projects: A dictionary of all projects with their respective term frequencies.
        """
        term_frequency = self.compute_term_frequency(projects)
        testing_project_vector = {}
        project_similarities = {}
        
        terms = projects[testing_pro]
        for term in terms.keys():
            tf_idf = self.compute_tf_idf(terms[term], len(projects), term_frequency[term])
            testing_project_vector[term] = tf_idf
        
        for training_project, training_terms in projects.items():
            if training_project != testing_pro:
                training_project_vector = {}
                
                for term in training_terms.keys():
                    tf_idf = self.compute_tf_idf(training_terms[term], len(projects), term_frequency.get(term, 0))
                    training_project_vector[term] = tf_idf
                
                similarity = self.compute_cosine_similarity(testing_project_vector, training_project_vector)
                project_similarities[training_project] = similarity

        sorted_similarities = dict(sorted(project_similarities.items(), key=lambda item: item[1], reverse=True))
        self.reader.write_similarity_scores(self.get_sim_dir(), testing_pro, sorted_similarities)

    def compute_jaccard_similarity(self, vector1, vector2):
        """
        Compute the Jaccard similarity between two binary vectors.
        
        :param vector1: A list or array representing the first binary vector.
        :param vector2: A list or array representing the second binary vector.
        :return: The Jaccard similarity score between the two vectors.
        """
        count = sum(1 for i, j in zip(vector1, vector2) if i == 1 and j == 1)
        # print(count, vector1, vector2)
        if len(vector1) == 0:
            return None
        return count / (2 * len(vector1) - count)

    def compute_cosine_similarity(self, v1, v2):
        """
        Compute the cosine similarity between two project vectors.
        
        :param v1: A dictionary representing the first vector.
        :param v2: A dictionary representing the second vector.
        :return: The cosine similarity score between the two vectors.
        """
        both_keys = set(v1.keys()).intersection(v2.keys())
        scalar = sum(v1[k] * v2[k] for k in both_keys)
        norm1 = math.sqrt(sum(f * f for f in v1.values()))
        norm2 = math.sqrt(sum(f * f for f in v2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return scalar / (norm1 * norm2)

    def compute_term_frequency(self, projects):
        """
        Compute the term-frequency map which stores, for every invocation,
        how many projects in the supplied list invoke it.

        :param projects: A dictionary where keys are project names and values are dictionaries of terms and their counts.
        :return: A dictionary where keys are terms and values are the number of projects invoking the term.
        """
        term_frequency = defaultdict(int)
        for terms in projects.values():
            for term in terms.keys():
                term_frequency[term] += 1

        return dict(term_frequency)

    def compute_tf_idf(self, count, total, freq):
        """
        Standard Term-Frequency Inverse Document Frequency (TF-IDF) calculation.

        :param count: The frequency of the term in the current project.
        :param total: The total number of projects.
        :param freq: The frequency of projects that include the term.
        :return: The TF-IDF value for the term.
        """
        if freq == 0:
            return 0.0
        return count * math.log(total / freq)
