import os

if os.name == 'nt': # on windows
    PLATTE_BASEPATH = "F:/programmieren/"
    PROJECTS_BASEPATH = "C:/Users/Florian/PycharmProjects/"
    FZI_DATASETS = ""

else: # on linux
    PLATTE_BASEPATH = "/media/florian/PLATTE/programmieren/"
    PROJECTS_BASEPATH = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/"
    FZI_DATASETS = "/disk/ml/datasets/"

SHOTS_FOLDER = PLATTE_BASEPATH + "VehicleReId/video_shots/"

#PATH_VEHICLEREID = "/disk/ml/datasets/VehicleReId/"
#PATH_CITYSCAPES = "/disk/ml/datasets/cityscapes/"
#PATH_BOXCARS = "/disk/ml/datasets/BoxCars21k/"

