# way to upload image
# way to save the image 
# function to make 
import os
import random
from flask import Flask
from flask import request
from flask import render_template
import dfdetector

"""Adapted from https://www.youtube.com/watch?v=BUh76-xD5qU"""

app = Flask(__name__)
UPLOAD_FOLDER = "./deepfake_detector/static/videos/"
DEVICE = "cpu"



@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            select = request.form.get('methodchoice')
            if image_file.filename.endswith(".jpg"):
                method, pred = dfdetector.DFDetector.detect_single(image_path=image_location, method=select)
            elif image_file.filename.endswith(".mp4") or image_file.filename.endswith(".avi"):
                method, pred = dfdetector.DFDetector.detect_single(video_path=image_location, method=select)
            # add random number to circumvent browser image caching of images with the same name
            randn = random.randint(0, 1000)
            return render_template("index.html", prediction = pred, method=method, image_loc=image_file.filename[:-4] + '.jpg', random_num = randn)
    return render_template("index.html", prediction = 0, method=None,image_loc = None,random_num=None)


if __name__ == "__main__":
    app.run(port=5000, debug=True)