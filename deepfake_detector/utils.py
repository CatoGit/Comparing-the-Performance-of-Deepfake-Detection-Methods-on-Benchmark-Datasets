import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import subprocess

def vidtimit_setup_real_videos(path_to_dataset):
    """
    Setting up the vidtimit dataset of real videos. The path for all real videos should be: ./vidtimitreal
    All videos should be in unzipped in their respective folders e.g. ./vidtimitreal/fadg0/
    The videos from the vidtimit.csv must be downloaded separately from http://conradsanderson.id.au/vidtimit/
    """
    # add jpg extension to all files
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            # give idx as new name, to avoid similar filenames in different folders
            os.rename(path + '/' + filename, path + '/' + filename + '.jpg')
    print("Creating .avi videos.")
    # create videos from jpgs
    file_ending = ".avi"
    counter = 0 
    os.chdir(path_to_dataset)
    for path, dirs, files in os.walk(path_to_dataset):
        if path == path_to_dataset:
            continue
        for d in dirs:
            if d != 'video':
                vid_path = os.path.join(path + '/' + d + '/%03d.jpg')
                vid_path2 = os.path.join(path + '/' + d + '/')
                # create real videos in avi format similar to deepfakes and avoid duplicate names with counter
                subprocess.call(['ffmpeg','-r', '25', '-i', f"{vid_path}","-c:v", "libxvid", f'{vid_path2 + str(counter)+  file_ending}'])
                counter += 1
    # remove all jpgs, so that only the videos are left
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            if filename.endswith(".jpg"):
                os.remove(path + '/' + filename)
