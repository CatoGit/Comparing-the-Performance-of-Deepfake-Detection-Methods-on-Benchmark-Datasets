## Benchmarking

To benchmark a method against a dataset, simply call .benchmark(dataset,data_path,method) on the DFDetector:

```benchmark_result = DFDetector.benchmark(dataset="uadfv",data_path="/home/jupyter/fake_videos", method="xception_celebdf")```

You can also benchmark from the command line by simply passing the three arguments via arg.parse:

#TODO

What should you specify for dataset, data_path and method?

### Data path setup:

Prior to benchmarking, you have to download the respective datasets and setup the path to the dataset. To get access to some of the datasets, the authors require you to fill out a form where you have to agree to their terms of use. After filling out the form, the datasets authors will send you a dataset download link. Links to the authors repositories, where you can access datasets or forms are linked below.
data_path takes arguments in the following way: your_path/datasetfolder
"your_path" is the path to the dataset folder (e.g. /home/jupyter/) and "datasetfolder" is the (unzipped) folder that contains the dataset (e.g. fake_videos). Below are the examples with the correct dataset folder names given:


| Benchmark dataset | Setup path | Download from |
| ------------- | ------------- | ------------- |
| uadfv  | your_path/fake_videos   | https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi |
| celebdf  | your_path/celebdf  |https://github.com/danmohaha/celeb-deepfakeforensics|
| dfdc  | your_path/   |https://www.kaggle.com/c/deepfake-detection-challenge/data|
| dftimithq | your_path/  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|
| dftimitlq  | your_path/  |Fake Videos: https://www.idiap.ch/dataset/deepfaketimit <br/> Real Videos: http://conradsanderson.id.au/vidtimit/|
| dfd  | your_path/ |https://github.com/ondyari/FaceForensics/|
| faceforensics++  | your_path/   |https://github.com/ondyari/FaceForensics/|


### Choices for methods:

There are 56 methods available for benchmarking. The dataset in the method name is the dataset that the method was fine-tuned on. 
If available, all methods made use of transfer learning (e.g. imagenet weights, noisy student weights), before they were fine-tuned for additional epochs on the respective dataset (see Experiments for more information).

| Deepfake detection methods | ACC on UADFV | ACC on Celeb-DF | ACC on DFDC| ACC on DF-1.0 | ACC on DF-Timit-HQ | ACC on DF-Timit-LQ| ACC on DFD | Accuracy on FF++|  
| ------------- | ------------- | ------------- |------------- |------------- | ------------- |------------- |------------- |------------- |
| xception_uadfv | a | b | c |  d | e | f | g | h |  
| efficientnetb7_uadfv |  |  |  |  |  |  |
| mesonet_uadfv |  |  |  |  |  |  |
| resnet_lstm_uadfv |  |  |  |  |  |  |
| efficientnetb1_lstm_uadfv |  |  |  |  |  |  |
| dfdcrank90_uadfv |  |  |  |  |  |  |
| full_ensemble_uadfv |  |  |  |  |  |  |
| xception_celebdf | a | b | c |  d | e | f | g | h |  
| efficientnetb7_celebdf |  |  |  |  |  |  |
| mesonet_celebdf |  |  |  |  |  |  |
| resnet_lstm_celebdf |  |  |  |  |  |  |
| efficientnetb1_lstm_celebdf |  |  |  |  |  |  |
| dfdcrank90_celebdf |  |  |  |  |  |  |
| full_ensemble_celebdf |  |  |  |  |  |  |
| xception_dfdc | a | b | c |  d | e | f | g | h |  
| efficientnetb7_dfdc |  |  |  |  |  |  |
| mesonet_dfdc |  |  |  |  |  |  |
| resnet_lstm_dfdc |  |  |  |  |  |  |
| efficientnetb1_lstm_dfdc |  |  |  |  |  |  |
| dfdcrank90_dfdc|  |  |  |  |  |  |
|full_ensemble_dfdc |  |  |  |  |  |  |
|xception_deeperforensics | a | b | c |  d | e | f | g | h |  
| efficientnetb7_deeperforensics |  |  |  |  |  |  |
| mesonet_deeperforensics |  |  |  |  |  |  |
| resnet_lstm_deeperforensics |  |  |  |  |  |  |
| efficientnetb1_lstm_deeperforensics|  |  |  |  |  |  |
| dfdcrank90_deeperforensics |  |  |  |  |  |  |
| full_ensemble_deeperforensics |  |  |  |  |  |  |
| xception_dftimithq| a | b | c |  d | e | f | g | h |  
| efficientnetb7_dftimithq|  |  |  |  |  |  |
| mesonet_dftimithq |  |  |  |  |  |  |
| resnet_lstm_dftimithq |  |  |  |  |  |  |
| efficientnetb1_lstm_dftimithq |  |  |  |  |  |  |
| dfdcrank90_dftimithq |  |  |  |  |  |  |
| full_ensemble_dftimithq|  |  |  |  |  |  |
| xception_dftimitlq| a | b | c |  d | e | f | g | h |  
| efficientnetb7_dftimitlq|  |  |  |  |  |  |
| mesonet_dftimitlq |  |  |  |  |  |  |
| resnet_lstm_dftimitlq |  |  |  |  |  |  |
| efficientnetb1_lstm_dftimitlq |  |  |  |  |  |  |
| dfdcrank90_dftimitlq |  |  |  |  |  |  |
| full_ensemble_dftimitlq|  |  |  |  |  |  |
| xception_dfd| a | b | c |  d | e | f | g | h |  
| efficientnetb7_dfd|  |  |  |  |  |  |
| mesonet_dfd|  |  |  |  |  |  |
| resnet_lstm_dfd |  |  |  |  |  |  |
| efficientnetb1_lstm_dfd |  |  |  |  |  |  |
| dfdcrank90_dfd |  |  |  |  |  |  |
| full_ensemble_dfd|  |  |  |  |  |  |
|xception_faceforensics++| a | b | c |  d | e | f | g | h |  
| efficientnetb7_faceforensics++|  |  |  |  |  |  |
| mesonet_faceforensics++|  |  |  |  |  |  |
| resnet_lstm_faceforensics++|  |  |  |  |  |  |
| efficientnetb1_lstm_faceforensics++|  |  |  |  |  |  |
| dfdcrank90_faceforensics++|  |  |  |  |  |  |
| full_ensemble_faceforensics++|  |  |  |  |  |  |









## Training

You can simply retrain the inference models yourself by calling train on the deepfake detector:

`DFDetector.train(dataset, data_path, method)`

Provide the datasets with their corresponding paths as well as the method in the same way as described in the "Inference" section.  
If you don't specify more arguments, the hyperparameters that were employed for the final models are used. Alternatively, further arguments can be given to the deepfake detector:

dataset = 'dataset_name'
data_path = 'dataset_path'
img_save_path = 'dataset_path'

`DFDetector.train(dataset, data_path, method, img_save_path=None, epochs=1, batch_size=32,
                     lr=0.001, folds=1, augmentation_strength='weak', fulltrain=False, faces_available=False)`
                  
`model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
                dataset="uadfv", data_path="/home/jupyter/fake_videos", method="xception",
                img_save_path="/home/jupyter/fake_videos",epochs=10, batch_size=32, lr=0.0001,folds=1,augmentation_strength="weak", fulltrain=True,faces_available=True,face_margin=0.3, seed=24)` 
                
`model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
        dataset="celebdf", data_path="/home/jupyter/celebdf", method="efficientnetb7",
        img_save_path="/home/jupyter/celebdf",epochs=10, batch_size=32, lr=0.0001,folds=1,augmentation_strength="weak", fulltrain=True,faces_available=True,face_margin=0.3, seed=24)`

Methods that can be used for training:

| Methods | Pretrained weights | 
| ------------- | ------------- | 
|xception| ImageNet|
|efficientnetb7| NoisyStudent|




Continue training from model (i.e. load model path): Not implemented (yet)

## Performance of Deepfake Detection Methods (Results)

The accuracy of the examined methods is presented here. Further insights can be found in the "Experiments"-file where the performance of each method is evaluated on other metrics as well.



