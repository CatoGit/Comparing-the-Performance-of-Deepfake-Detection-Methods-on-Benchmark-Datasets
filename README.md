[//]: # (Image References)
[webapp]: https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/webapp.PNG "webapp"
[dfdetect]: https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/efficientnetb7-on-all-datasets.png "dfdetect"

## Comparing the Performance of Deepfake Detection Methods on Benchmark Datasets

![Deepfake detection][dfdetect]

## Getting the models

Before predicting singles or benchmarking, the "weight" folder with the model checkpoints must be downloaded [here](https://drive.google.com/drive/u/0/folders/1C9T07evRE7S5rFa5H0SmdjCpLsR9Cqa4). After downloading it, copy the folder into:
```deepfake_detector/pretrained_mods/```

## Predict for a single video

The best way to detect a single deepfake video is to use the deepfake detection web application:
```python deepfake_detector/api.py``` 

![Wep application][webapp]

It utilizes the detect_single class method of the DFDetector class and provides an intuitive user interface. Alternatively the detect_single method can be called from the command line:

```python deepfake_detector/dfdetector.py --detect_single True --path_to_vid /example/path/0000_fake.mp4 --detection_method efficientnetb7_dfdc```

## Benchmarking

To benchmark a detection method on one of the five datasets, provide the path to the dataset and the desired detection method:

``` python deepfake_detector/dfdetector.py --benchmark True --data_path /example/path/fake_videos --detection_method efficientnetb7_dfdc```

A description of how the folders of the different datasets should be prepared is given below, and the arguments for the 35 available detection methods are given in the Section "Model Choices".

## Prepare the datasets

It is usually required to fill out a form to gain access to the datasets. After filling out the form, the datasets' authors will provide a dataset download link. The links to the author's repositories, where the access to the datasets can be requested, are below.
"your_path" is the path to the dataset folder (e.g. /home/jupyter/) and "datasetfolder" is the (unzipped) folder that contains the dataset (e.g. fake_videos). Examples are given below:

| Benchmark dataset keyword| Setup path | Download from |
| ------------- | ------------- | ------------- |
| uadfv  | your_path/fake_videos   | https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi |
| celebdf  | your_path/celebdf  |https://github.com/danmohaha/celeb-deepfakeforensics|
| dfdc  | your_path/dfdcdataset   |https://www.kaggle.com/c/deepfake-detection-challenge/data|
| dftimit_hq | your_path/DeepfakeTIMIT  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|
| dftimit_lq  | your_path/DeepfakeTIMIT  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|

The datasets should be extracted to the folders in the following way:

#### UADFV:
After extracting the fake_videos.zip folder, remove the file that are listed in uadfv_test.csv from the "fake" and "real" folders.
```
fake_videos/
├── fake
├── real

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

### Model choices:

The 35 deepfake detection methods that are available for benchmarking. The dataset in the detection method name is the dataset that the detection method was fine-tuned on. All detection methods made use of transfer learning (e.g. imagenet weights, noisy student weights) before they were fine-tuned for additional epochs on the respective dataset (see Experiments section in the thesis for more information).

The average performance of each detection method across all evaluated datasets is given for three metrics: the average accuracy, the average AUC and the average log(wP). 

| Deepfake detection method | Average Accuracy | Average AUC | Average log(wP)
| ------------- | ------------- | ------------- |------------- |
| xception_uadfv |  55.48|
| efficientnetb7_uadfv |57.12|
| mesonet_uadfv | 
| resnet_lstm_uadfv | 
| efficientnetb1_lstm_uadfv | 57.37|
| dfdcrank90_uadfv | 56.53|
| six_method_ensemble_uadfv | 58.26|
| xception_celebdf |  
| efficientnetb7_celebdf | 
| mesonet_celebdf | 
| resnet_lstm_celebdf | 59.87 
| efficientnetb1_lstm_celebdf | 
| dfdcrank90_celebdf | 
| six_method_ensemble_celebdf | 
| xception_dfdc | 
| efficientnetb7_dfdc | 
| mesonet_dfdc | 56.41|
| resnet_lstm_dfdc | 
| efficientnetb1_lstm_dfdc |
| dfdcrank90_dfdc| 
|six_method_ensemble_dfdc | 
| xception_dftimit_hq| 
| efficientnetb7_dftimit_hq| 63.47|
| mesonet_dftimit_hq | 
| resnet_lstm_dftimit_hq | 50.45|
| efficientnetb1_lstm_dftimit_hq | 63.31|
| dfdcrank90_dftimit_hq | 63.30|
| six_method_ensemble_dftimit_hq| 
| xception_dftimit_lq| 62.90|  
| efficientnetb7_dftimit_lq| 
| mesonet_dftimit_lq | 
| resnet_lstm_dftimitlq | 
| efficientnetb1_lstm_dftimit_lq | 
| dfdcrank90_dftimit_lq | 
| six_method_ensemble_dftimit_lq| 

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


