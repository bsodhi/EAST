import os
import sys
# Allows importing from: ../../ MUST COME AT THE TOP
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("In [{0}], appending [{1}] to system path.".format(__file__, parent))
sys.path.append(parent)


from sanic.response import json as json_response
from sanic import response as res
from sanic import Sanic
import asyncio
import json
import logging
import glob
import traceback
import argparse
import string
import random
import sqlite3
import concurrent.futures
import io
import tempfile
from datetime import datetime as DT
from pathlib import Path
import cv2
import numpy as np
import requests
from webapp import api

from PIL import Image
from vehicle_recognition.color.infer import infer as get_color
import matplotlib.pyplot as plt
from vehicle_recognition.vtype.predict import predict as get_vtype

logging.basicConfig(filename='backend.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%d-%m-%Y@%I:%M:%S %p')

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app = Sanic(__name__)
checkpoint_path = None
alpr_url = None


def random_str(size=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=size))


def ok_reply(body):
    return json_response({"status": "OK", "body": body})


def error_reply(body):
    return json_response({"status": "ERROR", "body": body})


def invoke_alpr_api(file_path, out_dir, task_type, cb_url):
    logging.info("Calling ALPR API for task {0} on file: {1}".format(
        task_type, file_path))
    data = {"task_dir": out_dir, "task_type": task_type,
            "callback_url": cb_url}
    res = requests.post(alpr_url, json=data)
    res.raise_for_status()
    jr = res.json()
    if jr["status"] != "OK":
        logging.error("API service failed to process request. "+jr["body"])
        return ("ERROR", jr["body"])
    else:
        logging.info("API service result: {0}".format(jr["body"]))
        return ("OK", jr["body"])


def process(task_type, file_path, cb_url):
    res = None
    try:
        task_dir = file_path[:-len("input_frame.jpg")]
        if task_type == "ocr":
            box_count = api.extract_text_boxes(
                file_path, task_dir, checkpoint_path)
            if box_count > 0:
                res = invoke_alpr_api(file_path, task_dir, task_type, cb_url)
            else:
                res = ("ERROR", "No textbox found in image.")
                logging.info("No text box found in file: {}".format(file_path))

        elif task_type == "alpr":
            res = invoke_alpr_api(file_path, task_dir, task_type, cb_url)

        elif task_type == "color":
            color = get_color(Image.open(file_path))
            res = ("OK", color)

        elif task_type == "vtype":
            vtype = get_vtype(plt.imread(file_path))
            res = ("OK", vtype)

        else:
            msg = "Invalid task type supplied: {}".format(task_type)
            res = ("ERROR", msg)
            logging.error(msg)

    except Exception as ex:
        res = ("ERROR", str(ex))
        logging.exception(
            "Error occurred when processing image {}".format(file_path))
    return res


@app.route("/api", methods=["POST",])
async def api_handler(req):
    if not req.json:
        return error_reply("Expected JSON request.")

    task_type = req.json.get("task_type")
    file_path = req.json.get("file_path")
    cb_url = req.json.get("callback_url")

    if not task_type or not file_path or not cb_url:
        return error_reply("Required arguments are missing. Found: "
                           + str(req.json))

    logging.info("Processing {}".format(req.json))
    status, msg = process(task_type, file_path, cb_url)
    if status == "ERROR":
        return error_reply(msg)
    else:
        # vehicle_recognition
        if task_type == "color" or task_type == "vtype":
            return ok_reply(msg)
        else:
            return ok_reply("Request successfully queued.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Server port number.")
    parser.add_argument("workers", type=int, help="No. of worker instances.")
    parser.add_argument("alpr_url", type=str, help="ALPR service URL.")
    parser.add_argument("checkpoint_path", type=str,
                        help="Checkpoint data folder path.")

    parser.add_argument("-s", "--host", type=str, dest="host",
                        default="0.0.0.0", help="Server port number.")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    alpr_url = args.alpr_url
    logging.info("Starting the service with args: {}".format(vars(args)))
    app.run(host=args.host, port=args.port, workers=args.workers)
