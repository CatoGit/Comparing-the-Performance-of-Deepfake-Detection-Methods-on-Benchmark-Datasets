from dfdetector import DFDetector
if __name__ == "__main__":
    # results = DFDetector.benchmark(
    #     dataset="uadfv", data_path='C:/Users/Chris/Desktop/fake_videos', method="xception")
    model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
        dataset="uadfv", data_path='C:/Users/Chris/Desktop/fake_videos', method="xception",
        img_save_path="C:/Users/Chris/Desktop/fake_videos",epochs=1, batch_size=32, lr=0.001, faces_available=True)