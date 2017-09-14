import os, glob
import numpy as np
import cv2


def chose_from_folder(folder_path, file_extension="*", missing_parameter=None):
    """
    :param folder_path: str path to the folder that shall be examined
    :param file_extension: filter for such files, e.g. '*.hdf5',
    :param missing_parameter: give an additional explanation how to avoid this chooser
    :return: the folder/file name, attention it doesn't add a "/" for folders in the end.
    """
    if folder_path[-1] != "/":
        folder_path += "/"
    if missing_parameter:
        print("The parameter", missing_parameter, "was not set")
    print("Pick a suitable element from the folder:", folder_path, "cwd:", os.getcwd())
    folder_list = sorted(glob.glob(folder_path + file_extension))
    for idx, folder_content in enumerate(folder_list):
        print("[{}] {}".format(idx, folder_content))
    return str(folder_list[int(input("Enter number: "))])

#
# def getFramesVRI():
#     return os.system('ffmpeg -i /data/mlprak1/VehicleReId-Untouched/video_shots/1A.mov '
#                      '/fzi/ids/mlprak1/no_backup/VehicleReId/1A/1A_%06d.png')


def draw_annotations(img, coords, data_format="3d_reg", fac="n/a", numbers=True):
    img = np.copy(img)
    # reformat it from zero centered to  3x (0-255)
    # minimum = np.amin(img)
    # maximum = np.amax(img)
    # img = ((img - minimum + 0) * 255 / (maximum - minimum)).astype(np.uint8).copy()

    if data_format == "3d_reg":
        np.array(coords).reshape((1, 20))

        # FORMAT of coords, but the the x and y are sorted.
        #   0     1     2    3
        # "x1", "y1", "x2", "y2",
        #        4:red             4+8: red               5: yellow          5+8: yellow
        # "top_front_outer_x", "top_front_outer_y", "top_front_inner_x", "top_front_inner_y",
        # "top_back_inner_x", "top_back_inner_y", "top_back_outer_x", "top_back_outer_y",
        # "bot_front_outer_x", "bot_front_outer_y", "bot_front_inner_x", "bot_front_inner_y",
        # "bot_back_inner_x", "bot_back_inner_y", "bot_back_outer_x", "bot_back_outer_y",

        colors = [(0, 0, 255),  # red           0
                  (0, 255, 255),  # yellow      1
                  (255, 255, 255),  # white     2
                  (255, 255, 0),  # cyan        3
                  (255, 0, 0),  # blue          4
                  (0, 0, 0),  # black           5
                  (0, 255, 0),  # green         6
                  (255, 0, 255),  # magenta     7
                  ]
        # ################### MEANING OF THE COLORS/ANNOTATIONS ##########################
        # The cars are always in this angle
        #
        #       (0,0) >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  (x,0)
        #         v
        #         v        cyan  ~~~~~~~~~~
        #         v        /                ~~~~~~~~~~ red
        #         v   white  ~~~~~~~~~~               /  |
        #         v     |              ~~~~~~~~~~ yellow |
        #         v     |                           |    |
        #         v     |                           |  f |
        #         v     |  purple                   |  r |
        #         v     | /                         |   blue
        #         v   green  ~~~~~~~~~              |   /
        #         v                    ~~~~~~~~~~~~black
        #       (0,y)

        for i in range(8):
            img = cv2.circle(img, (int(coords[4 + i]), int(coords[4 + 8 + i])), 10, color=colors[i], thickness=9)
            if numbers:
                img = cv2.putText(img, str(i), (int(coords[4 + i]), int(coords[4 + 8 + i])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 4)
        if numbers:
            img = cv2.line(img, (int(coords[5]), int(coords[5+8])), (int(coords[8]), int(coords[8+8])), (255, 255, 255), thickness=5)
            img = cv2.line(img, (int(coords[4]), int(coords[4 + 8])), (int(coords[9]), int(coords[9 + 8])), (255, 255, 255), thickness=5)

        img = cv2.putText(img, fac, (int(coords[0]), int(coords[1])-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 23, 23), 4)
        img = cv2.putText(img, fac, (int(coords[0]), int(coords[3])+5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 23, 23), 4)

    else:  # Mode: Merge Areas
        (_, x1, y1, x2, y2, classname) = coords
        if classname == 'Front':
            color = (255, 255, 0)
            c = 'c'
        elif classname == 'Back':
            color = (255, 255, 255)
            c = 'w'
        elif classname == 'Side':
            color = (0, 255, 0)
            c = 'b'
        elif classname == 'Outer':
            color = (255, 0, 0)
            c = 'g'
        elif classname == 'Top':
            color = (255, 0, 225)
            c = 'm'

        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=5)

        # ################### MEANING OF THE COLORS/ANNOTATIONS ##########################
        # The cars are always in this angle
        #
        #       (0,0) >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  (x,0)
        #         v
        #         v     O   x  ~~~~~~~~~~~~              O
        #         v        /                ~~~~~~~~F~~~ F
        #         v     S ~~~~~~~~~~~~              S /f |
        #         v     |              ~~~~~~~~~~~~ x  r |
        #         v     |                           |  o |
        #         v     |           side            |  n |
        #         v     |   x                       |  t |
        #         v     | /                         |    x
        #         v     x ~~~~~~~~~~~~              |  /
        #         v     OS             ~~~~~~~~~~~~FS    OF
        #       (0,y)
        #
        #    In my first step it only seems necessary to teach the net to some sides of this cube
        #    FRONTBB: four point marked "F"
        #    SIDEBB:  four points marked "S"
        #    OUTERBB: four points marked "O"

    return img


def show_img_data(img, img_data, im_size, outpath="./", prefix="anno_", scale=True):
    make_missing_dirs(outpath)
    width, height = img_data['width'], img_data['height']
    (resized_width, resized_height) = get_new_img_size(width, height, im_size)
    w_ratio = resized_width / float(width)
    h_ratio = resized_height / float(height)
    for bbox in img_data['bboxes']:
        # simple parser inserts in this format:
        # {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2),
        #  'bb_x1': int(bb_x1), 'bb_y1': int(bb_y1), 'bb_x2': int(bb_x2), 'bb_y2': int(bb_y2),
        #  'bb_x3': int(bb_x3), 'bb_y3': int(bb_y3), 'bb_x4': int(bb_x4), 'bb_y4': int(bb_y4),
        #  'bb_x5': int(bb_x5), 'bb_y5': int(bb_y5), 'bb_x6': int(bb_x6), 'bb_y6': int(bb_y6),
        #  'bb_x7': int(bb_x7), 'bb_y7': int(bb_y7), 'bb_x8': int(bb_x8), 'bb_y8': int(bb_y8),
        #  },

        order = [
            'x1', 'y1', 'x2', 'y2',
            'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8',
            'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8'
        ]
        if scale:
            ratio_multiplier = [w_ratio, h_ratio]*2 + [w_ratio]*8 + [h_ratio]*8
        else:
            ratio_multiplier = [1]*20
        coords = np.array([int(bbox[order[i]]*ratio_multiplier[i]) for i in range(20)])
        img = draw_annotations(img, coords, data_format="3d_reg", fac=bbox['class'])
        if 'crop' in img_data:
            img = cv2.putText(img, str(img_data['crop']),
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 4)
    print("Showing File", img_data['filepath'])
    filename = prefix + img_data['filepath'][img_data['filepath'].rfind("/")+1:]

    cv2.imwrite(outpath + filename, img)
    print(img_data)
    # input("saved annotations to " + filename)
    return


def show_net_input(X, img_data, C, outpath="./"):
    img = np.copy(X)[0][:, :, [2, 1, 0]]
    # reformat it from zero centered to  3x (0-255)
    minimum = np.amin(img)
    maximum = np.amax(img)
    img = ((img - minimum + 0) * 255 / (maximum - minimum)).astype(np.uint8).copy()
    show_img_data(img, img_data, C.im_size, outpath)


class Logger:
    def __init__(self, log_dir, log_file, eol="\n"):
        self.log_path = log_dir + log_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.eol = eol

    def log(self, s):
        with open(self.log_path, "a") as log:
            log.write(str(s)+self.eol)

    def log_print(self, *args):
        print(*args)
        with open(self.log_path, "a") as log:
            log.write(" ".join(map(str, args))+"\n")


def make_missing_dirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height



def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape
    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # print("ratio=", ratio)

    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2, bb3d):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    real_bb3d = [int(round(v // ratio)) for v in bb3d]

    return (real_x1, real_y1, real_x2, real_y2, real_bb3d)
