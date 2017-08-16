import os, glob

def chose_from_folder(folder_path, file_extension="*", missing_parameter=None):
    """
    :param folder_path: str path to the folder that shall be examined
    :param file_extension: filter for such files, e.g. '*.hdf5',
    :param missing_parameter: give an additional explanation how to avoid this chooser
    :return: the folder/file name, attention it doesn't add a "/" for folders in the end.
    """
    assert folder_path[-1] == "/"
    if missing_parameter:
        print("The parameter", missing_parameter, "was not set")
    print("Pick a suitable element from the folder:", folder_path, "cwd:", os.getcwd())
    folder_list = sorted(glob.glob(folder_path + file_extension))
    for idx, folder_content in enumerate(folder_list):
        print("[{}] {}".format(idx, folder_content))
    return str(folder_list[int(input("Enter number: "))])
#
# def getFramesVRI():
#     return os.system('ffmpeg -i /data/mlprak1/VehicleReId-Untouched/video_shots/1A.mov /fzi/ids/mlprak1/no_backup/VehicleReId/1A/1A_%06d.png')
