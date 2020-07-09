[//]: # (Image References)
[webapp]: https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/webapp.PNG "webapp"

## Getting the models

Before predicting singles or benchmarking, the model checkpoints need to be downloaded. All 35 models can be downloaded here: #TODO

## Predict on a single image or video

The best way to make predictions on a single image or video is to use the deepfake detection web application (api.py). 

![Wep application][webapp]

It utilizes the detect_single class method of the DFDetector class and provides an intuitive user interface. Alternatively the detect_single method can be used in code, for example:

```method, result = DFDetector.detect_single(video_path="/example/path/video.mp4", image_path=None, method="xception_uadfv")```

## Benchmarking

To benchmark a method against one of the five datasets, simply call .benchmark(dataset,data_path,method) on the DFDetector:

```benchmark_result = DFDetector.benchmark(dataset="uadfv",data_path="/home/jupyter/fake_videos", method="xception_celebdf")```

You can also benchmark from the command line by simply passing the three arguments via arg.parse:

#TODO

What should you specify for dataset, data_path and method?

## Prepare the datasets

Prior to benchmarking, you have to download the respective datasets and setup the path to each dataset. To get access to some of the datasets, the datasets' authors require you to fill out a form where you have to agree to their terms of use. After filling out the form, the datasets' authors will send you a dataset download link. Links to the authors repositories, where you can access datasets or forms are linked below.
data_path takes arguments in the following way: your_path/datasetfolder
"your_path" is the path to the dataset folder (e.g. /home/jupyter/) and "datasetfolder" is the (unzipped) folder that contains the dataset (e.g. fake_videos). Below are the examples with the correct dataset folder names given:


| Benchmark dataset keyword| Setup path | Download from |
| ------------- | ------------- | ------------- |
| uadfv  | your_path/fake_videos   | https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi |
| celebdf  | your_path/celebdf  |https://github.com/danmohaha/celeb-deepfakeforensics|
| dfdc  | your_path/dfdcdataset   |https://www.kaggle.com/c/deepfake-detection-challenge/data|
| dftimit_hq | your_path/DeepfakeTIMIT  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|
| dftimit_lq  | your_path/DeepfakeTIMIT  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|

The datasets should be extracted to the folders in the following way.

#### UADFV:
After extracting the fake_videos.zip folder, remove the file that are listed in uadfv_test.csv from the "fake" and "real" folders.
```
fake_videos/
├── fake
├── real
├── test
```
#### Celeb-DF:
```
celebdf/
├── YouTube-real
├── List_of_testing_videos.txt
├── Celeb-synthesis
├── Celeb-real
```
#### DF-TIMIT-LQ and DF-TIMIT-HQ:
The vidtimit dataset of real videos is only available in frame format. These frames need to be turned into videos. utils.py provides a method vidtimit_setup_real_videos(path_to_dataset) that turns the frames into the necessary videos. Afterwards, the folders should be ordered like this:
```
DeepfakeTIMIT/
├── lower_quality
├── higher_quality
├── dftimitreal
```
#### DFDC:
Video files from folders 0 to 44 must be copied to the train folder. Video files from folders 45 to 49 must be copied to the test folder. They can be downloaded from Kaggle using the following command for each folder: !wget --load-cookies cookies.txt https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_45.zip and the cookies.txt file.
```
dfdcdataset/
├── train
├── test
├── val
```


### Choices for methods:

There are 35 methods available for benchmarking. The dataset in the method name is the dataset that the method was fine-tuned on. 
If available, all methods made use of transfer learning (e.g. imagenet weights, noisy student weights), before they were fine-tuned for additional epochs on the respective dataset (see Experiments section in the thesis for more information).

| Deepfake detection methods | ACC on UADFV | ACC on Celeb-DF | ACC on DFDC| ACC on DF-TIMIT-HQ | ACC on DF-TIMIT-LQ|
| ------------- | ------------- | ------------- |------------- |------------- | ------------- |
| xception_uadfv | **100.00** | 37.07 | 50.00 |  44.17 | 45.83 | 
| efficientnetb7_uadfv | **100.00** | 35.33 | 49.90 |  50.00 | 50.00 |
| mesonet_uadfv | 89.29 | 65.25 | 55.70 |  70.00 | 77.50 |
| resnet_lstm_uadfv | **100.00**  | 49.23 | 56.30 |  55.92 | 61.67 | 
| efficientnetb1_lstm_uadfv | **100.00** | 38.03 | 55.10 |  41.67 | 51.67 |
| dfdcrank90_uadfv | **100.00** | 35.52 | 50.45 | 49.17 | 47.50 |
| six_method_ensemble_uadfv | **100.00** | 38.80 | 51.66 |  44.17 | 56.67 |
| xception_celebdf | **100.00** | 98.07 | 53.80 |  51.67 | 83.33 |  
| efficientnetb7_celebdf | **100.00** | 97.68 | 52.40 | 50.00 | 60.00 |
| mesonet_celebdf | 64.29 | 78.19 | 54.50 | 65.00 | 87.50 |
| resnet_lstm_celebdf | 71.43 | 95.37 | 56.10 | 50.83 | 70.00 |
| efficientnetb1_lstm_celebdf | 78.57 | 97.68 | 54.00 | 61.67 | 80.00 | 
| dfdcrank90_celebdf | 96.43 | 98.65 | 53.17 | 55.00 |  83.33 |
| six_method_ensemble_celebdf | 89.29 | **99.04** |  53.27 | 50.83 | 80.83 |
| xception_dfdc | 78.57 | 55.99 | 90.50 |  72.50 | 95.00 |  
| efficientnetb7_dfdc | 89.29 | 67.18 | **93.60** | 90.00 |  99.17 |
| mesonet_dfdc | 63.13 | 58.40 | 47.50 | 48.33 |  
| resnet_lstm_dfdc | 50.00 | 66.41 | 70.80 | 55.83 |  55.83 |
| efficientnetb1_lstm_dfdc | 78.57 | 67.18 | 87.90 | 96.68 |  99.17 |
| dfdcrank90_dfdc| 82.14 | 59.14 | 91.94 | 77.50 |  **100.00** |
|six_method_ensemble_dfdc | 82.14 | 58.98 | 92.95 | 94.17 |  **100.00** |
| xception_dftimithq| 57.14 | 72.01 | 50.90 |  91.67 | 96.67 | 
| efficientnetb7_dftimithq| 35.71 | 35.14 | 46.70 | **100.00** | **100.00** |
| mesonet_dftimithq | 50.00 | 64.87 | 51.10 | 72.50 | 89.17 |
| resnet_lstm_dftimithq | 46.43 | 64.87 | 51.10 | 72.50 |  
| efficientnetb1_lstm_dftimithq | 32.14 | 35.33 | 50.40 | 99.17 |  
| dfdcrank90_dftimithq | 25.00 | 37.84 | 53.68 | **100.00** | 
| six_method_ensemble_dftimithq| 39.29 | 51.93 | 52.17 | 99.17 |
| xception_dftimitlq| a | b | c |  d | e |   
| efficientnetb7_dftimitlq|  |  |  |  | 
| mesonet_dftimitlq |  |  |  |  |  
| resnet_lstm_dftimitlq |  |  |  |  | 
| efficientnetb1_lstm_dftimitlq |  |  |  |  |  
| dfdcrank90_dftimitlq |  |  |  |  | 
| six_method_ensemble_dftimitlq|  |  |  |  |  

## Training

You can simply retrain the inference models yourself by calling train_method on the deepfake detector. An example for training the xception method on the UADFV dataset.

`model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
                dataset="uadfv", data_path="/home/jupyter/fake_videos", method="xception",
                img_save_path="/home/jupyter/fake_videos",epochs=10, batch_size=32, lr=0.0001,folds=1,augmentation_strength="weak", fulltrain=True,faces_available=False,face_margin=0.3, seed=24)` 

Provide the datasets with their corresponding paths as well as the method in the same way as described in the "Inference" section (Note: for the DFDC dataset you need to download the remaining folders 0-44 to train it).  
Hyperparameter arguments can be given to the deepfake detector. The arguments that were used to train the final models are given in the hyperparameter settings section of the experiments file.
                  
Method arguments that can be used for training:

| Method argument | Pretrained weights | 
| ------------- | ------------- | 
|xception| ImageNet|
|efficientnetb7| NoisyStudent|
|mesonet|Mesonet|
|resnet_lstm| ImageNet|
|efficientnetb1_lstm| ImageNet|

## Performance of Deepfake Detection Methods (Results)

The accuracy of the examined methods is presented here. Further insights can be found in the "Experiments"-file where the performance of each method is evaluated on other metrics as well.


