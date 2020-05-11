from dfdetector import DFDetector
import os
if __name__ == "__main__":
    
    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    #     dataset="uadfv", data_path='/home/jupyter/fake_videos', method="efficientnetb7",
    #     img_save_path="/home/jupyter/fake_videos",epochs=10, batch_size=8, lr=0.001,folds=5,augmentation_strength="weak", fulltrain=False,faces_available=True,face_margin=0, seed=24)

    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    # dataset="celebdf", data_path='C:/Users/Chris/Desktop/Celeb-DF-v2', method="xception",
    # img_save_path="C:/Users/Chris/Desktop/Celeb-DF-v2",epochs=1, batch_size=32, lr=0.001, faces_available=False, face_margin=0)


# img save path and data path are the same -> redundancy can be removed!



#result = DFDetector.detect(video=video_file, method="xception", heatmap=False)

    benchmark_result = DFDetector.benchmark(dataset="uadfv",data_path="C:/Users/Chris/Desktop/fake_videos", method="efficientnetb7")

# ->compare: whether to compare with the results of other methods that were precomputed by myself