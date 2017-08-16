import os

def chose_from_folder(folder_path, missing_parameter=None):
    assert folder_path[-1] == "/"
    if missing_parameter:
        print("The parameter", missing_parameter, "was not set")
    print("Pick a suitable element from the folder:", folder_path, "cwd:", os.getcwd())
    folder_list = sorted(os.listdir(folder_path))
    for idx, folder_content in enumerate(folder_list):
        print("[{}] {}".format(idx, folder_content))
    return folder_path + str(folder_list[int(input("Enter number: "))])
#
# def getFramesVRI():
#     return os.system('ffmpeg -i /data/mlprak1/VehicleReId-Untouched/video_shots/1A.mov /fzi/ids/mlprak1/no_backup/VehicleReId/1A/1A_%06d.png')
