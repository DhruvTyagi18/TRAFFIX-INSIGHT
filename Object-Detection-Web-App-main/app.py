from ultralytics import YOLO
import time
import numpy as np


import cv2
from flask import Flask, render_template, request, Response, session, redirect, url_for

from flask_socketio import SocketIO
import yt_dlp as youtube_dl

import os
from werkzeug.utils import secure_filename

model_object_detection = YOLO("yolov8n.pt")

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
stop_flag = False

class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        print ("*********************************Video Streaming******************************")
        # self.VIDEO = cv2.VideoCapture(0)
        # self.VIDEO.set(10, 200)
        self._preview = False
        self._flipH = False
        self._detect = False
        self._model = False
        self._confidence = 75.0

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = int(value)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)


    def show(self, url):
        print(url)
        self._preview = False
        self._flipH = False
        self._detect = False

        self._confidence = 75.0
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best",
            "forceurl": True,
        }
        # Create a youtube-dl object
        ydl = youtube_dl.YoutubeDL(ydl_opts)

        # Extract the video URL
        info = ydl.extract_info(url, download=False)
        url = info["url"]

        cap = cv2.VideoCapture(url)
        while True:
            if self._preview:
                if stop_flag:
                    print("Process Stopped")
                    return

                grabbed, frame = cap.read()
                if not grabbed:
                    break
                if self.flipH:
                    frame = cv2.flip(frame, 1)
                if self.detect:
                    print(self._confidence)

                    # frame = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
                    # frame = cv2.resize(frame, (500,500
                    # ))
                    # Detect objects
                    results = model_object_detection.predict(frame, conf=self._confidence/100)

                    frame, labels = results[0].plot()
                    list_labels = []
                    # labels_confidences
                    for label in labels:
                        confidence = label.split(" ")[-1]
                        label = (label.split(" "))[:-1]
                        label = " ".join(label)
                        list_labels.append(label)
                        list_labels.append(confidence)
                        socketio.emit('label', list_labels)
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.imencode(".jpg", frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                snap = np.zeros((
                    1000,
                    1000
                ), np.uint8)
                label = "Streaming Off"
                H, W = snap.shape
                font = cv2.FONT_HERSHEY_PLAIN
                color = (255, 255, 255)
                cv2.putText(snap, label, (W//2 - 100, H//2),
                            font, 2, color, 2)
                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# check_settings()
VIDEO = VideoStreaming()
@app.route('/image')
def upload_form():
    return render_template('image.html')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('show_image', filename=filename))

# @app.route('/analyze_image', methods=['POST'])
# def analyze_image():
#     image_path = request.form.get('image_path')
#     if not image_path:
#         return 'No image provided for analysis', 400

#     # Perform object detection on the uploaded image
#     image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], image_path))
#     results = model_object_detection.predict(image, conf=VIDEO.confidence / 100)

#     # Process detection results
#     detected_objects = []
#     for result in results:
#         if result.names:
#             detected_objects.extend(result.names)

#     # Get predicted classes for detected objects
#     predicted_classes = [get_predicted_class(obj) for obj in detected_objects]

#     analyzed_image_path = image_path

#     # Redirect to the image analysis result page
#     return render_template('imageindex.html', image_path=image_path, analyzed_image_path=analyzed_image_path, detected_objects=detected_objects, predicted_classes=predicted_classes)

# def get_predicted_class(object_name):
#     # Here you can implement the logic to map the detected object name to its predicted class
#     # For example, you can use a dictionary lookup or some other method based on your model
#     # Replace this with your actual implementation
#     return object_name  # Placeholder for demonstration, replace it with the actual implementation
from flask import render_template
def get_predicted_class(object_name):
    # Mapping detected object names to class names
    class_mapping = {
        "person": "Person",
        "dog": "Dog",
        "car":"Car",
        "vehicle":"Vehicle",
        # Add more mappings as needed
    }
    # Check if the object name exists in the mapping, otherwise return "Unknown"
    return class_mapping.get(object_name, "Car")

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    image_path = request.form.get('image_path')
    if not image_path:
        return 'No image provided for analysis', 400

    # Perform object detection on the uploaded image
    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], image_path))
    results = model_object_detection.predict(image, conf=VIDEO.confidence / 100)

    # Process detection results
    detected_objects = []
    for result in results:
        if result.names:
            detected_objects.extend(result.names)

    # Get predicted classes for detected objects
    predicted_classes = [get_predicted_class(obj) for obj in detected_objects]

    analyzed_image_path = image_path

    # Render the template and pass detected_objects, predicted_classes, and get_predicted_class function
    return render_template('imageindex.html', image_path=image_path, analyzed_image_path=analyzed_image_path, detected_objects=detected_objects, predicted_classes=predicted_classes, get_predicted_class=get_predicted_class)



@app.route('/showimage/<filename>')
def show_image(filename):
    return render_template('imageindex.html', image_path=filename)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('hompage.html')

@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    print("index")
    global stop_flag
    stop_flag = False
    if request.method == 'POST':
        print("Index post request")
        url = request.form['url']
        print("index: ", url)
        session['url'] = url
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    url = session.get('url', None)
    print("video feed: ", url)
    if url is None:
        return redirect(url_for('homepage'))
    return Response(VIDEO.show(url), mimetype='multipart/x-mixed-replace; boundary=frame')

# * Button requests
@app.route("/request_preview_switch")
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    print("*"*10, VIDEO.preview)
    return "nothing"

@app.route("/request_flipH_switch")
def request_flipH_switch():
    VIDEO.flipH = not VIDEO.flipH
    print("*"*10, VIDEO.flipH)
    return "nothing"

@app.route("/request_run_model_switch")
def request_run_model_switch():
    VIDEO.detect = not VIDEO.detect
    print("*"*10, VIDEO.detect)
    return "nothing"


@app.route('/update_slider_value', methods=['POST'])
def update_slider_value():
    slider_value = request.form['sliderValue']
    VIDEO.confidence = slider_value
    return 'OK'

@app.route('/stop_process')
def stop_process():
    print("Process stop Request")
    global stop_flag
    stop_flag = True
    return 'Process Stop Request'

@socketio.on('connect')
def test_connect():
    print('Connected')

if __name__ == "__main__":
    socketio.run(app, debug=True)
