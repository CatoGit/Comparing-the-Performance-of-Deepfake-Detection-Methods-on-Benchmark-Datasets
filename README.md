## Training

You can simply retrain the inference models yourself by calling train on the deepfake detector:

`DFDetector.train(dataset, data_path, method)`

Provide the datasets with their corresponding paths as well as the method in the same way as described in the "Inference" section.  
If you don't specify more arguments, the hyperparameters that were employed for the final models are used. Alternatively, further arguments can be given to the deepfake detector:

`DFDetector.train(dataset, data_path, method, img_save_path=None, epochs=1, batch_size=32,
                     lr=0.001, folds=1, augmentation_strength='weak', fulltrain=False, faces_available=False)`
                     
             
