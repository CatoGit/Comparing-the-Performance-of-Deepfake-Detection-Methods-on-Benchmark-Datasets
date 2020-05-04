# result = dfdetector.detect(video, method, optional->facedetector->default retinafce) --> detects whether file is real/fake
# result = dfdetector.benchmark(dataset,method) ->benchmarks method on datasets test data


from deepfake_detector import datasets

class DFDetector():
    def __init__(self, facedetector="retinaface_resnet", visuals=False):
        self.facedetector = facedetector
        self.visuals = visuals

    @classmethod   
    def detect(cls, video=None,  method="xception"):
        return result

    @classmethod
    def benchmark(cls, dataset=None, method="xception"):
        cls.dataset=dataset
        cls.method=method
        datasets.clf.dataset(img_dir, df,img_size, augmentations_weak)
        return benchmark_result

#result = DFDetector.detect(video=video_file, method="xception", heatmap=False)

#benchmark_result = DFDetector.benchmark(dataset="uadfv", method="xception", compare=False)

#->compare: whether to compare with the results of other methods that were precomputed by myself