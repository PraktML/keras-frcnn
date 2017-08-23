import os

# there are two global variable names that can be referred to. There are the root directories of the
# $BOXCARS116K_PATH$, $VRI_SHOTS_PATH$, the method below replaces those with the applicable path.

if os.name == 'nt': # on windows
    BOXCARS116K_PATH = "F:/programmieren/BoxCars116k/images/"
    BOXCARS116K_JSON_FILE = BOXCARS116K_PATH + "json_data/dataset.json"
    VRI_SHOTS_PATH = "F:/programmieren/VehicleReId/video_shots/"

elif os.uname()[1]=="florian-ultra" and os.path.exists("/media/florian/PLATTE/"):  # Florian + external drive
    BOXCARS116K_PATH = "/media/florian/PLATTE/programmieren/BoxCars116k/images/"
    BOXCARS116K_JSON_FILE = "/media/florian/PLATTE/programmieren/BoxCars116k/json_data/dataset.json"
    VRI_SHOTS_PATH = "/media/florian/PLATTE/programmieren/VehicleReId/video_shots/"

elif os.uname()[1]=="florian-ultra":
    BOXCARS116K_PATH = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/BoxCars116k/"
    BOXCARS116K_JSON_FILE = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/BoxCars116k/json_data/dataset.json"
    VRI_SHOTS_PATH = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/vri_shots/"

elif os.uname()[1]=="patrick": #TODO: hier den Pfad anpassen.
    BOXCARS116K_PATH = "/home/patrick/MLPrakt/Data/BoxCars116k/images/"
    BOXCARS116K_JSON_FILE = BOXCARS116K_PATH + "json_data/dataset.json"
    VRI_SHOTS_PATH = "/home/patrick/MLPrakt/Data/video_shots/"

else:  # os.uname()[1]=='ids-graham': # at FZI
    BOXCARS116K_PATH = "/disk/ml/datasets/BoxCars116k/data/images/"
    VRI_SHOTS_PATH = "/data/mlprak1/VehicleReId/video_shots/"
    BOXCARS116K_JSON_FILE = "/disk/ml/datasets/BoxCars116k/data/json_data/dataset.json"

FRAMES_VRI = [
    {"name": "1A", "frames": [1862, 2496, 3016],
     "offset": 0,  "sep_y": 0, "sep_m": 0.43, "from":0, "to": 20000},
    {"name": "1B", "frames": [1800, 7402, 10300],
     "offset": -2, "sep_y": 10, "sep_m": 0.22, "from":0, "to": 20000},
    {"name": "2A", "frames": [1800, 7402, 12278, 12240, 12030],
     "offset": 0, "sep_y": -20, "sep_m": 0.50, "from":0, "to": 20000},
    {"name": "2B", "frames": [1862, 4390, 9270, 9476, 9910],
     "offset": -2, "sep_y": -10, "sep_m": 0.38, "from":0, "to": 20000},
    {"name": "3A", "frames": [1862, 922, 4896, 9388],
     "offset": 0, "sep_y": -65, "sep_m": 0.67, "from":0, "to": 20000},
    {"name": "3B", "frames": [1862, 922, 4896],
     "offset": -2, "sep_y": -10, "sep_m": 0.4, "from":0, "to": 8000},  # dieser offset stimmt ned
    # {"name": "4A", "frames": [806, 4390,8410, 8800,],
    #  "offset": +1, "sep_y": -100, "sep_m": 0.65 , "from":0, "to": 8000},
    # {"name": "4B", "frames": [806, 4390, 7934, 8166],
    #  "offset": -4, "sep_y": 6, "sep_m": 0.35, "from":0, "to": 8000},
    # {"name": "5A", "frames": [804, 4390,17902, 18014],
    #  "offset": -3, "sep_y": -250, "sep_m": 0.60, "from":0, "to": 8000},
    # {"name": "5B", "frames": [804, 4390, 14238, 14372],
    #  "offset": -2, "sep_y": -35, "sep_m": 0.4, "from":0, "to": 8000},
]


def variable_path_to_abs(path, boxcars116k_path=BOXCARS116K_PATH, vri_shots_path=VRI_SHOTS_PATH):
    path = path.replace("$BOXCARS116K_PATH$", boxcars116k_path)
    path = path.replace("$VRI_SHOTS_PATH$", vri_shots_path)
    return os.path.normpath(path)