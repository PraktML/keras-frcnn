import os

if os.name == 'nt': # on windows
    PLATTE_BASEPATH = "F:/programmieren/"
    PROJECTS_BASEPATH = "C:/Users/Florian/PycharmProjects/"
    FZI_DATASETS = ""

else: # on linux
    PLATTE_BASEPATH = "/media/florian/PLATTE/programmieren/"
    PROJECTS_BASEPATH = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/"
    FZI_DATASETS = "/disk/ml/datasets/"
