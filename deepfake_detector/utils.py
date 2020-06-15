import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import subprocess
import pandas as pd
import numpy as np


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
                subprocess.call(
                    ['ffmpeg', '-r', '25', '-i', f"{vid_path}", "-c:v", "libxvid", f'{vid_path2 + str(counter)+  file_ending}'])
                counter += 1
    # remove all jpgs, so that only the videos are left
    for path, dirs, files in os.walk(path_to_dataset):
        for filename in files:
            if filename.endswith(".jpg"):
                os.remove(path + '/' + filename)


def dfdc_metadata_setup():
    """Returns training, testing and validation video meta data frames for the DFDC dataset."""
    #read in metadata
    print("Reading metadata...this can take a minute.")
    df_train0 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata0.json')
    df_train0.loc[df_train0.shape[0]] = 0
    df_train0.rename({3: 'folder'}, axis='index', inplace=True)
    df_train1 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata1.json')
    df_train1.loc[df_train1.shape[0]] = 1
    df_train1.rename({3: 'folder'}, axis='index', inplace=True)
    df_train2 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata3.json')
    df_train2.loc[df_train2.shape[0]] = 3
    df_train2.rename({3: 'folder'}, axis='index', inplace=True)
    df_train3 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata4.json')
    df_train3.loc[df_train3.shape[0]] = 4
    df_train3.rename({3: 'folder'}, axis='index', inplace=True)
    df_train4 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata5.json')
    df_train4.loc[df_train4.shape[0]] = 5
    df_train4.rename({3: 'folder'}, axis='index', inplace=True)
    df_train5 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata6.json')
    df_train5.loc[df_train5.shape[0]] = 6
    df_train5.rename({3: 'folder'}, axis='index', inplace=True)
    df_train6 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata8.json')
    df_train6.loc[df_train6.shape[0]] = 8
    df_train6.rename({3: 'folder'}, axis='index', inplace=True)
    df_train7 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata9.json')
    df_train7.loc[df_train7.shape[0]] = 9
    df_train7.rename({3: 'folder'}, axis='index', inplace=True)
    df_train8 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata10.json')
    df_train8.loc[df_train8.shape[0]] = 10
    df_train8.rename({3: 'folder'}, axis='index', inplace=True)
    df_train9 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata12.json')
    df_train9.loc[df_train9.shape[0]] = 12
    df_train9.rename({3: 'folder'}, axis='index', inplace=True)
    df_train10 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata13.json')
    df_train10.loc[df_train10.shape[0]] = 13
    df_train10.rename({3: 'folder'}, axis='index', inplace=True)
    df_train11 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata14.json')
    df_train11.loc[df_train11.shape[0]] = 14
    df_train11.rename({3: 'folder'}, axis='index', inplace=True)
    df_train12 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata15.json')
    df_train12.loc[df_train12.shape[0]] = 15
    df_train12.rename({3: 'folder'}, axis='index', inplace=True)
    df_train13 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata16.json')
    df_train13.loc[df_train13.shape[0]] = 16
    df_train13.rename({3: 'folder'}, axis='index', inplace=True)
    df_train14 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata17.json')
    df_train14.loc[df_train14.shape[0]] = 17
    df_train14.rename({3: 'folder'}, axis='index', inplace=True)
    df_train15 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata19.json')
    df_train15.loc[df_train15.shape[0]] = 19
    df_train15.rename({3: 'folder'}, axis='index', inplace=True)
    df_train16 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata20.json')
    df_train16.loc[df_train16.shape[0]] = 20
    df_train16.rename({3: 'folder'}, axis='index', inplace=True)
    df_train17 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata22.json')
    df_train17.loc[df_train17.shape[0]] = 22
    df_train17.rename({3: 'folder'}, axis='index', inplace=True)
    df_train18 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata23.json')
    df_train18.loc[df_train18.shape[0]] = 23
    df_train18.rename({3: 'folder'}, axis='index', inplace=True)
    df_train19 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata24.json')
    df_train19.loc[df_train19.shape[0]] = 24
    df_train19.rename({3: 'folder'}, axis='index', inplace=True)
    df_train20 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata25.json')
    df_train20.loc[df_train20.shape[0]] = 25
    df_train20.rename({3: 'folder'}, axis='index', inplace=True)
    df_train21 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata26.json')
    df_train21.loc[df_train21.shape[0]] = 26
    df_train21.rename({3: 'folder'}, axis='index', inplace=True)
    df_train39 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata27.json')
    df_train39.loc[df_train39.shape[0]] = 27
    df_train39.rename({3: 'folder'}, axis='index', inplace=True)
    df_train22 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata29.json')
    df_train22.loc[df_train22.shape[0]] = 29
    df_train22.rename({3: 'folder'}, axis='index', inplace=True)
    df_train23 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata30.json')
    df_train23.loc[df_train23.shape[0]] = 30
    df_train23.rename({3: 'folder'}, axis='index', inplace=True)
    df_train24 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata31.json')
    df_train24.loc[df_train24.shape[0]] = 31
    df_train24.rename({3: 'folder'}, axis='index', inplace=True)
    df_train25 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata32.json')
    df_train25.loc[df_train25.shape[0]] = 32
    df_train25.rename({3: 'folder'}, axis='index', inplace=True)
    df_train26 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata34.json')
    df_train26.loc[df_train26.shape[0]] = 34
    df_train26.rename({3: 'folder'}, axis='index', inplace=True)
    df_train27 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata35.json')
    df_train27.loc[df_train27.shape[0]] = 35
    df_train27.rename({3: 'folder'}, axis='index', inplace=True)
    df_train28 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata36.json')
    df_train28.loc[df_train28.shape[0]] = 36
    df_train28.rename({3: 'folder'}, axis='index', inplace=True)
    df_train29 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata37.json')
    df_train29.loc[df_train29.shape[0]] = 37
    df_train29.rename({3: 'folder'}, axis='index', inplace=True)
    df_train30 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata38.json')
    df_train30.loc[df_train30.shape[0]] = 38
    df_train30.rename({3: 'folder'}, axis='index', inplace=True)
    df_train31 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata40.json')
    df_train31.loc[df_train31.shape[0]] = 40
    df_train31.rename({3: 'folder'}, axis='index', inplace=True)
    df_train32 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata41.json')
    df_train32.loc[df_train32.shape[0]] = 41
    df_train32.rename({3: 'folder'}, axis='index', inplace=True)
    df_train33 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata43.json')
    df_train33.loc[df_train33.shape[0]] = 43
    df_train33.rename({3: 'folder'}, axis='index', inplace=True)
    df_train34 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata44.json')
    df_train34.loc[df_train34.shape[0]] = 44
    df_train34.rename({3: 'folder'}, axis='index', inplace=True)
    df_train35 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata45.json')
    df_train35.loc[df_train35.shape[0]] = 45
    df_train35.rename({3: 'folder'}, axis='index', inplace=True)
    df_train36 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata46.json')
    df_train36.loc[df_train36.shape[0]] = 46
    df_train36.rename({3: 'folder'}, axis='index', inplace=True)
    df_train37 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata47.json')
    df_train37.loc[df_train37.shape[0]] = 47
    df_train37.rename({3: 'folder'}, axis='index', inplace=True)
    df_train38 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata49.json')
    df_train38.loc[df_train38.shape[0]] = 49
    df_train38.rename({3: 'folder'}, axis='index', inplace=True)
    df_train40 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata18.json')
    df_train40.loc[df_train40.shape[0]] = 18
    df_train40.rename({3: 'folder'}, axis='index', inplace=True)
    df_train41 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata2.json')
    df_train41.loc[df_train41.shape[0]] = 2
    df_train41.rename({3: 'folder'}, axis='index', inplace=True)
    df_train42 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata28.json')
    df_train42.loc[df_train42.shape[0]] = 28
    df_train42.rename({3: 'folder'}, axis='index', inplace=True)
    df_train43 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata42.json')
    df_train43.loc[df_train43.shape[0]] = 42
    df_train43.rename({3: 'folder'}, axis='index', inplace=True)
    df_train44 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata7.json')
    df_train44.loc[df_train44.shape[0]] = 7
    df_train44.rename({3: 'folder'}, axis='index', inplace=True)
    df_train45 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata11.json')
    df_train45.loc[df_train45.shape[0]] = 11
    df_train45.rename({3: 'folder'}, axis='index', inplace=True)
    df_train46 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata21.json')
    df_train46.loc[df_train46.shape[0]] = 21
    df_train46.rename({3: 'folder'}, axis='index', inplace=True)
    df_train47 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata33.json')
    df_train47.loc[df_train47.shape[0]] = 33
    df_train47.rename({3: 'folder'}, axis='index', inplace=True)
    df_train48 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata39.json')
    df_train48.loc[df_train48.shape[0]] = 39
    df_train48.rename({3: 'folder'}, axis='index', inplace=True)
    df_train49 = pd.read_json(os.getcwd() + '/deepfake_detector/data/metadata/metadata48.json')
    df_train49.loc[df_train49.shape[0]] = 48
    df_train49.rename({3: 'folder'}, axis='index', inplace=True)
    # combine metadata
    print("Formatting metadata...")
    df_train = [df_train0, df_train1, df_train2, df_train3, df_train4,
                df_train5, df_train6, df_train7, df_train8, df_train9, df_train10,
                df_train11, df_train12, df_train13, df_train14, df_train15,
                df_train16, df_train17, df_train18, df_train19, df_train20, df_train21,
                df_train22, df_train23, df_train24, df_train25, df_train26,
                df_train27, df_train28, df_train29, df_train30, df_train31, df_train32,
                df_train33, df_train34, df_train35, df_train36, df_train37,
                df_train38, df_train39, df_train40, df_train41, df_train42, df_train43, df_train44,
                df_train45, df_train46, df_train47, df_train48, df_train49]
    all_meta = pd.concat(df_train, axis=1)
    all_meta = all_meta.T  # transpose
    all_meta['video'] = all_meta.index  # create video column from index
    all_meta.reset_index(drop=True, inplace=True)  # drop index
    del all_meta['split']
    # recode labels
    all_meta['label'] = all_meta['label'].apply(
        lambda x: 0 if x == 'REAL' else 1)
    del all_meta['original']
    # sample 16974 fakes from 45 folders -> that's approx. 378 fakes per folder
    train_df = all_meta[all_meta['folder'] < 45]
    # 16974 reals in train data and 89629 fakes
    reals = train_df[train_df['label'] == 0]
    #del reals['folder']
    reals['folder']
    fakes = train_df[train_df['label'] == 1]
    fakes_sampled = fakes[fakes['folder'] == 0].sample(378, random_state=24)
    # sample the same number of fake videos from every folder
    for num in range(45):
        if num == 0:
            continue
        sample = fakes[fakes['folder'] == num].sample(378, random_state=24)
        fakes_sampled = fakes_sampled.append(sample, ignore_index=True)
    # drop 36 videos randomly to have exactly 16974 fakes
    np.random.seed(24)
    drop_indices = np.random.choice(fakes_sampled.index, 36, replace=False)
    fakes_sampled = fakes_sampled.drop(drop_indices)
    #del fakes_sampled['folder']
    fakes_sampled['folder']
    all_meta_train = pd.concat([reals, fakes_sampled], ignore_index=True)
    # get 1000 samples from training data that are used for margin and augmentation validation
    real_sample = all_meta_train[all_meta_train['label'] == 0].sample(
        300, random_state=24)
    fake_sample = all_meta_train[all_meta_train['label'] == 1].sample(
        300, random_state=24)
    full_margin_aug_val = real_sample.append(fake_sample, ignore_index=True)
    # create test set
    test_df = all_meta[all_meta['folder'] > 44]
    del test_df['folder']
    all_meta_test = test_df.reset_index(drop=True)

    return all_meta_train, all_meta_test, full_margin_aug_val
