from dataReader import *
import os

class SuccessCalculator:

    def __init__(self, src_dir, sub_folder, testing_start_pos, testing_end_pos):
        self.reader = DataReader()  # Create an instance of DataReader
        self.src_dir = src_dir
        self.sub_folder = sub_folder
        self.rec_dir = os.path.join(self.src_dir, self.sub_folder, "Recommendations")
        self.gt_dir = os.path.join(self.src_dir, self.sub_folder, "GroundTruth")
        self.testing_start_pos = testing_start_pos
        self.testing_end_pos = testing_end_pos

    def compute_success_rate(self, n):
        testing_projects_id = self.reader.read_project_list(
            os.path.join(self.src_dir, "List.txt"), 
            self.testing_start_pos, 
            self.testing_end_pos)

        number_of_matches = 0
        for project in testing_projects_id.values():
            top_rec = self.reader.read_recommendation_file(os.path.join(self.rec_dir, project), n)
            ground_truth = self.reader.read_ground_truth_invocations(os.path.join(self.gt_dir, project))
            
            intersection = ground_truth.intersection(top_rec)
            if intersection: number_of_matches += 1
            
        return (number_of_matches / len(testing_projects_id)) * 100

    def compute_precision(self, n):
        testing_projects_id = self.reader.read_project_list(
            os.path.join(self.src_dir, "List.txt"), 
            self.testing_start_pos, 
            self.testing_end_pos
        )

        precision = 0
        for project in testing_projects_id.values():
            top_rec = self.reader.read_recommendation_file(os.path.join(self.rec_dir, project), n)
            ground_truth = self.reader.read_ground_truth_invocations(os.path.join(self.gt_dir, project))

            intersection = ground_truth.intersection(top_rec)
            precision += len(intersection) / n

        return precision / len(testing_projects_id)
    
    def compute_recall(self, n):
        testing_projects_id = self.reader.read_project_list(
            os.path.join(self.src_dir, "List.txt"), 
            self.testing_start_pos, 
            self.testing_end_pos
        )

        recall = 0
        for project in testing_projects_id.values():
            top_rec = self.reader.read_recommendation_file(os.path.join(self.rec_dir, project), n)
            ground_truth = self.reader.read_ground_truth_invocations(os.path.join(self.gt_dir, project))

            intersection = ground_truth.intersection(top_rec)
            recall += len(intersection) / len(ground_truth)

        return recall / len(testing_projects_id)

# def main():
#     # Modify these paths and positions according to your file structure
#     src_dir = "/home/smanduru/ReCS/MemoRec/dataset/pkg_cls_curated_RQ2/"
#     sub_folder = "evaluation/round1"
#     testing_start_pos = 0
#     testing_end_pos = 10

#     calculator = SuccessCalculator(src_dir, sub_folder, testing_start_pos, testing_end_pos)

#     n = 5  # Number of recommendations to consider

#     success_rate = calculator.compute_success_rate(n)
#     print(f"Success Rate: {success_rate}%")

#     precision = calculator.compute_precision(n)
#     print(f"Precision: {precision}")

#     recall = calculator.compute_recall(n)
#     print(f"Recall: {recall}")


# if __name__ == "__main__":
#     main()