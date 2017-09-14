import scripts.helper as helper
import scripts.settings as settings
import pickle

run_folder = helper.chose_from_folder("runs/", "*", "") + "/"

# anno_file = helper.chose_from_folder("annotations/", "*.txt", "")
anno_file = helper.chose_from_folder(run_folder, "*.txt", "")

out_file = run_folder + "anno_test.txt"

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
            # unfortunately the paths were saved like this in the splits file!
            boxcars116k_path="/disk/ml/datasets/BoxCars116k/data/images/",
            vri_shots_path="/data/mlprak1/VehicleReId/video_shots/"
        )
        if idx % (num_lines//100) == 0:
            print(idx, filename_splits)
        if splits[filename_splits] == "test":
            output += line
        idx += 1
with open(out_file, 'w') as f:
    f.write(output)