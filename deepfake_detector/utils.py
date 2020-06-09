import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def vidtimit_setup_real_videos(path_to_dataset):
    """
    Setting up the vidtimit dataset of real videos. The path for all real videos should be: ./vidtimitreal
    All videos should be in unzipped in their respective folders e.g. ./vidtimitreal/fadg0/
    All videos from the vidtimit.csv must be downloaded from http://conradsanderson.id.au/vidtimit/
    """
    # add jpg extension to all files
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            os.rename(path + '/' + filename, path + '/' + filename + '.jpg')
    print("Creating .avi videos.")
    # create videos from jpgs
    for path, dirs, files in os.walk(path_to_dataset):
        if path == path_to_dataset:
            continue
        for d in dirs:
            if d != 'video':
                print(d)
                vid_path = os.path.join(path + '/' + d + '/')
                # create real videos in avi format similar to deepfakes
                ! ffmpeg - r 25 - i {vid_path + "%03d.jpg"} - c: v libxvid  {vid_path + d + ".avi"}

    # remove all jpgs, so that only the videos are left
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            if filename.endswith(".jpg"):
                os.remove(path + '/' + filename)
