## Predict on a single image or video

The best way to make predictions on a single image or video is to use the deepfake detection web application (api.py). It utilizes the detect_single class method of the DFDetector class and provides an intuitive user interface. Alternatively the detect_single method can be used in code, for example:

```method, result = DFDetector.detect_single(video_path="/example/path/video.mp4", image_path=None, method="xception_uadfv")```

## Benchmarking

To benchmark a method against one of the five datasets, simply call .benchmark(dataset,data_path,method) on the DFDetector:

```benchmark_result = DFDetector.benchmark(dataset="uadfv",data_path="/home/jupyter/fake_videos", method="xception_celebdf")```

You can also benchmark from the command line by simply passing the three arguments via arg.parse:

#TODO

What should you specify for dataset, data_path and method?

### Data path setup:

Prior to benchmarking, you have to download the respective datasets and setup the path to each dataset. To get access to some of the datasets, the datasets' authors require you to fill out a form where you have to agree to their terms of use. After filling out the form, the datasets' authors will send you a dataset download link. Links to the authors repositories, where you can access datasets or forms are linked below.
data_path takes arguments in the following way: your_path/datasetfolder
"your_path" is the path to the dataset folder (e.g. /home/jupyter/) and "datasetfolder" is the (unzipped) folder that contains the dataset (e.g. fake_videos). Below are the examples with the correct dataset folder names given:


| Benchmark dataset | Setup path | Download from |
| ------------- | ------------- | ------------- |
| uadfv  | your_path/fake_videos   | https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi |
| celebdf  | your_path/celebdf  |https://github.com/danmohaha/celeb-deepfakeforensics|
| dfdc  | your_path/dfdcdataset   |https://www.kaggle.com/c/deepfake-detection-challenge/data|
| dftimithq | your_path/DeepfakeTIMIT  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|
| dftimitlq  | your_path/DeepfakeTIMIT  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|

Additional setup information:
dfdc: The folder dfdcdataset contains the five video folders dfdc_train_part_45 up to part 49. They can be downloaded from Kaggle using the following command for each folder: !wget --load-cookies cookies.txt https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_45.zip and the cookies.txt file.
dftimithq & dftimitlq: The vidtimit dataset of real videos is only available in frame format. These frames need to be turned into videos. utils.py provides a method vidtimit_setup_real_videos(path_to_dataset) that turns the frames into the necessary videos. Alternatively the preprocessed dataset can be downloaded from: 

### Choices for methods:

There are 35 methods available for benchmarking. The dataset in the method name is the dataset that the method was fine-tuned on. 
If available, all methods made use of transfer learning (e.g. imagenet weights, noisy student weights), before they were fine-tuned for additional epochs on the respective dataset (see Experiments section in the thesis for more information).

| Deepfake detection methods | ACC on UADFV | ACC on Celeb-DF | ACC on DFDC| ACC on DF-Timit-HQ | ACC on DF-Timit-LQ|
| ------------- | ------------- | ------------- |------------- |------------- | ------------- |
| xception_uadfv | a | b | c |  d | e | 
| efficientnetb7_uadfv |  |  |  |  
| mesonet_uadfv |  |  |  |  
| resnet_lstm_uadfv |  |  |  |  
| efficientnetb1_lstm_uadfv |  |  |  |  
| dfdcrank90_uadfv |  |  |  | 
| full_ensemble_uadfv |  |  |  |  
| xception_celebdf | a | b | c |  d | e |  
| efficientnetb7_celebdf |  |  |  | 
| mesonet_celebdf |  |  |  |  |
| resnet_lstm_celebdf |  |  |  |  | 
| efficientnetb1_lstm_celebdf |  |  |  |  |  
| dfdcrank90_celebdf |  |  |  |  |  
| full_ensemble_celebdf |  |  |  |  | 
| xception_dfdc | a | b | c |  d | e |  
| efficientnetb7_dfdc |  |  |  |  |  
| mesonet_dfdc |  |  |  |  |  
| resnet_lstm_dfdc |  |  |  |  |  
| efficientnetb1_lstm_dfdc |  |  |  |  |  
| dfdcrank90_dfdc|  |  |  |  |  
|full_ensemble_dfdc |  |  |  |  |  
| xception_dftimithq| a | b | c |  d | e | 
| efficientnetb7_dftimithq|  |  |  |  | 
| mesonet_dftimithq |  |  |  |  | 
| resnet_lstm_dftimithq |  |  |  |  |  
| efficientnetb1_lstm_dftimithq |  |  |  |  |  
| dfdcrank90_dftimithq |  |  |  |  | 
| full_ensemble_dftimithq|  |  |  |  |  
| xception_dftimitlq| a | b | c |  d | e |   
| efficientnetb7_dftimitlq|  |  |  |  | 
| mesonet_dftimitlq |  |  |  |  |  
| resnet_lstm_dftimitlq |  |  |  |  | 
| efficientnetb1_lstm_dftimitlq |  |  |  |  |  
| dfdcrank90_dftimitlq |  |  |  |  | 
| full_ensemble_dftimitlq|  |  |  |  |  

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



