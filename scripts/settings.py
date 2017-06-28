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
BOXCARS_FOLDER = PLATTE_BASEPATH + "BoxCars116k/"
#PATH_VEHICLEREID = "/disk/ml/datasets/VehicleReId/"
#PATH_CITYSCAPES = "/disk/ml/datasets/cityscapes/"
#PATH_BOXCARS = "/disk/ml/datasets/BoxCars21k/"

FRAMES_VRI = [
    {"name": "1A", "frames": [1862, 2496, 3016],                "offset": 0,  "sep_y": 0, "sep_m": 0.43, "from":0, "to": 20000},
    {"name": "1B", "frames": [1800, 7402, 10300],               "offset": -2, "sep_y": 10, "sep_m": 0.22, "from":0, "to": 20000},
    {"name": "2A", "frames": [1800, 7402, 12278, 12240, 12030], "offset": 0, "sep_y": -20, "sep_m": 0.50, "from":0, "to": 20000},
    {"name": "2B", "frames": [1862, 4390, 9270, 9476, 9910],    "offset": -2, "sep_y": -10, "sep_m": 0.38, "from":0, "to": 20000},
    {"name": "3A", "frames": [1862, 922, 4896, 9388],           "offset": 0, "sep_y": -65, "sep_m": 0.67, "from":0, "to": 20000},
                                                                ### dieser offset stimmt ned
    {"name": "3B", "frames": [1862, 922, 4896],                 "offset": -2, "sep_y": -10, "sep_m": 0.4, "from":0, "to": 8000},
    {"name": "4A", "frames": [806, 4390,8410, 8800,],           "offset": +1, "sep_y": -100, "sep_m": 0.65 , "from":0, "to": 8000},
#    {"name": "4B", "frames": [806, 4390, 7934, 8166],           "offset": -4, "sep_y": 6, "sep_m": 0.35, "from":0, "to": 8000},
#    {"name": "5A", "frames": [804, 4390,17902, 18014],          "offset": -3, "sep_y": -250, "sep_m": 0.60, "from":0, "to": 8000},
#    {"name": "5B", "frames": [804, 4390, 14238, 14372],         "offset": -2, "sep_y": -35, "sep_m": 0.4, "from":0, "to": 8000},

]
FRAMES_VRI = []