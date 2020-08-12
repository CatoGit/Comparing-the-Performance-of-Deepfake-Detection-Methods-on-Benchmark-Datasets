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

A description of how the folders of the different datasets should be prepared is given below, and the arguments for the 35 available detection methods are given in the Section "Performance of Deepfake Detection Methods" in column "Deepfake Detection Method".

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

### Performance of Deepfake Detection Methods (Results) 

The 35 deepfake detection methods that are available for benchmarking. The dataset in the detection method name is the dataset that the detection method was fine-tuned on. All detection methods made use of transfer learning (e.g. imagenet weights, noisy student weights) before they were fine-tuned for additional epochs on the respective dataset (see Experiments section in the thesis for more information).

The average performance of each detection method across all evaluated datasets is given for three metrics: the average accuracy, the average AUC and the average log(wP). 

|Nr.| Deepfake Detection Method | Average Accuracy | Average AUC | Average log(wP), R=0.9|
| -------------| ------------- | ------------- | ------------- |------------- |
|1 |xception_uadfv |  55.48|56.59|-3,56|
| 2|efficientnetb7_uadfv |57.12|47.17|-3,56|
|3 |mesonet_uadfv | 71.63|77.10|-3,57|
| 4|resnet_lstm_uadfv | 63.99|69.12|-3,37|
|5 |efficientnetb1_lstm_uadfv | 57.37|58.67|-3,44|
| 6|dfdcrank90_uadfv | 56.53|57.84|-3,57|
|7 |six_method_ensemble_uadfv | 58.26|65.36|-3,43|
| 8|xception_celebdf |  77.45|81.41|-2,63|
|9 |efficientnetb7_celebdf |72.09| 82.21|-2,58|
|10 |mesonet_celebdf | 69.97|81.85|-3,50|
|11 |resnet_lstm_celebdf | 68.83|68.87|-3,63|
|12 |efficientnetb1_lstm_celebdf | 74.46|91.37|-2,18|
| 13|dfdcrank90_celebdf | 77.32|84.27|-2,49|
| 14|six_method_ensemble_celebdf | 74.65|88.76|-2,33|
| 15|xception_dfdc | 78.64|90.72|-2,60|
| 16|efficientnetb7_dfdc | **87.98**|94.60|-2.16|
| 17|mesonet_dfdc | 56.41|64.35|-4,40|
| 18|resnet_lstm_dfdc | 59.87|68.87|-4,09|
| 19|efficientnetb1_lstm_dfdc |86.02|**94.79**|-1,80|
| 20|dfdcrank90_dfdc| 82.14|94.21|-1.91|
|21|six_method_ensemble_dfdc |85.65|94.40|**-1,62**|
|22 |xception_dftimit_hq| 73.75|79.52|-3,00|
| 23|efficientnetb7_dftimit_hq| 63.47|66.30|-2,63|
| 24|mesonet_dftimit_hq | 65.60|70.31|-3,89|
|25 |resnet_lstm_dftimit_hq | 50.45|43.35|-4,53|
|26 |efficientnetb1_lstm_dftimit_hq | 63.31|72.82|-2,86|
|27 |dfdcrank90_dftimit_hq | 63.30|72.90|-2,52|
| 28|six_method_ensemble_dftimit_hq| 68.35|72.75|-2,54|
|29 |xception_dftimit_lq| 62.90| 73.80| -2,57|
| 30|efficientnetb7_dftimit_lq| 64.76|79.77|-2,51|
| 31|mesonet_dftimit_lq | 66.49 |76.80|-3,16|
|32 |resnet_lstm_dftimit_lq |65.82|75.03|-3.46|
| 33|efficientnetb1_lstm_dftimit_lq | 67.66|66,13|-2,84|
|34 |dfdcrank90_dftimit_lq | 67.83|70.72|-2,60|
|35 |six_method_ensemble_dftimit_lq| 65.19|75.33|-2,57|

## Training

The detection methods can be re-trained by calling the train_method on the deepfake detector. Below is an example for training the `xception model` type on the UADFV dataset:

`python deepfake_detector/dfdetector.py --train True --model_type xception --dataset uadfv --save_path your_path/fake_videos --data_path your_path/fake_videos ` 

Provide the datasets with their corresponding paths as well as the model type in the same way as described below (Note: for the DFDC dataset you need to download the remaining folders 0-44).  
Hyperparameter arguments can be given to the deepfake detector. The arguments that were used to train the final deepfake detection methods are given in the hyperparameter settings section of the experiments .xlsx file that is available in the `data` folder.
                  
Modet lype arguments that can be used for training:

| Model Type | Pretrained Weights | 
| ------------- | ------------- | 
|xception| ImageNet|
|efficientnetb7| NoisyStudent|
|mesonet|Mesonet|
|resnet_lstm| ImageNet|
|efficientnetb1_lstm| ImageNet|




