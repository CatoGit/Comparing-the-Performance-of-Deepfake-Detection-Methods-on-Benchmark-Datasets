from dfdetector import DFDetector
if __name__ == "__main__":
    # results = DFDetector.benchmark(
    #     dataset="uadfv", data_path='C:/Users/Chris/Desktop/fake_videos', method="xception")
    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    #     dataset="uadfv", data_path='C:/Users/Chris/Desktop/fake_videos', method="xception",
    #     img_save_path="C:/Users/Chris/Desktop/fake_videos",epochs=1, batch_size=32, lr=0.001, faces_available=True)

    model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
        dataset="uadfv", data_path='C:/Users/Chris/Desktop/fake_videos', method="efficientnetb7",
        img_save_path="C:/Users/Chris/Desktop/fake_videos",epochs=1, batch_size=32, lr=0.001, faces_available=True)

#result = DFDetector.detect(video=video_file, method="xception", heatmap=False)

#benchmark_result = DFDetector.benchmark(dataset="uadfv", method="xception", compare=False)

# ->compare: whether to compare with the results of other methods that were precomputed by myself