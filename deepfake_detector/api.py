# way to upload image
# way to save the image 
# function to make 
import os
from flask import Flask
from flask import request
from flask import render_template
import dfdetector

"""Adapted from https://www.youtube.com/watch?v=BUh76-xD5qU"""

app = Flask(__name__)
UPLOAD_FOLDER = "./deepfake_detector/static/videos/"
DEVICE = "cpu"
MODEL = "xception_uadfv"



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
            pred = dfdetector.DFDetector.detect_single(image_path=image_location, method=MODEL)
            print(pred)
            return render_template("index.html", prediction = pred, image_loc=image_file.filename)
    return render_template("index.html", prediction = 0, image_loc = None)

if __name__ == "__main__":
    app.run(port=12000, debug= True)