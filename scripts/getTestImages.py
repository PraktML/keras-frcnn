import scripts.helper as helper
import scripts.settings as settings
import pickle

run_folder = helper.chose_from_folder("runs/", "*", "") + "/"

anno_file = helper.chose_from_folder("annotations/", "*.txt", "")

out_file = "annotations/testfile.txt"

with open(run_folder + "splits.pickle", 'rb') as splits_f:
    splits = pickle.load(splits_f)

num_lines = sum(1 for _ in open(anno_file, 'r'))

with open(anno_file, 'r') as f:
    idx = 0
    output = ""
    for line in f:
        line_split = line.strip().split(',')
        filename = line_split[0]

        filename_splits = settings.variable_path_to_abs(
            filename,
            boxcars116k_path="/disk/ml/datasets/BoxCars116k/data/images/",
            vri_shots_path="/data/mlprak1/VehicleReId/video_shots/"
        )
        print (filename_splits)
        if splits[filename_splits] == "test":
            output += line
with open(out_file, 'w') as f:
    f.write(output)