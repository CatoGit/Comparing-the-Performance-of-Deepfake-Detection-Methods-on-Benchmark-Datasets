from deepfake_detector.models import xception
import os 
if __name__ == "__main__":
    print("Hallo")
    cwd = os.getcwd()
    print(cwd)
    model = xception.imagenet_pretrained_xception()
    print(model)