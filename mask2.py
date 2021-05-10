import io
import os
import sys
import time
import yaml
import cv2
import os
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.adapters import classify
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference




cap = cv2.VideoCapture(0)
cur_dir: str = sys.path[0]
CONFIG_PATH: str = os.path.join(cur_dir, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    operational_config = yaml.safe_load(f)
if "device" not in operational_config:
    raise Exception("Failed to load configuration")

face_model = operational_config["models"]["face_detection"]["model"]
face_threshold = operational_config["models"]["face_detection"]["threshold"]
face_labels = operational_config["models"]["face_detection"]["labels"]
mask_model = operational_config["models"]["mask_classifier"]["model"]
mask_labels = operational_config["models"]["mask_classifier"]["labels"]
mask_threshold = operational_config["models"]["mask_classifier"]["threshold"]
deployment: dict = operational_config["deployment"]

print('Loading {} with {} labels.'.format(face_model, face_labels))
interpreter = make_interpreter(face_model)
interpreter.allocate_tensors()
labels = read_label_file(face_labels)
inference_size = input_size(interpreter)


# Apply mask / no mask classifier
mask_interpreter = make_interpreter(mask_model)
mask_interpreter.allocate_tensors()
mask_labels = read_label_file(mask_labels)
input_details = mask_interpreter.get_input_details()
output_details = mask_interpreter.get_output_details()


while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
	cv2_im = frame
	cv2_im_rgb1 = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
	cv2_im_rgb = cv2.resize(cv2_im_rgb1, inference_size)
	run_inference(interpreter, cv2_im_rgb.tobytes())
	objs = get_objects(interpreter, face_threshold)[:2]
	height, width, channels = cv2_im.shape
	scale_x, scale_y = width / inference_size[0], height / inference_size[1]
	for obj in objs:
		bbox = obj.bbox.scale(scale_x, scale_y)


		x0, y0 = int(bbox.xmin), int(bbox.ymin)
		x1, y1 = int(bbox.xmax), int(bbox.ymax)
		percent = int(100 * obj.score)
		label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

		cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
		cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
				             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

		shall_raise_alert = False
		
		if  labels.get(obj.id, obj.id) != "UNMASKED":
			break
		proba = obj.score
		if proba < mask_threshold:
			break
		shall_raise_alert = True
		print(
			"Alert: no mask with probability {:08.6f}: {}".format(
			    percent, labels.get(obj.id, obj.id)
			)
		)

		if not shall_raise_alert:
			print(f"No alerts to raise. Proba ({percent}) is below alert threshold ({mask_threshold}\n")

	cv2.imshow('frame', cv2_im)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
