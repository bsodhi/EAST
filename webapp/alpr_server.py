import requests
import numpy as np
import cv2
from flask import Flask, abort, session, redirect, url_for, request, render_template
from functools import wraps
from markupsafe import escape
from passlib.hash import pbkdf2_sha256
from pathlib import Path
from flask.helpers import flash, send_file, send_from_directory
from datetime import datetime as DT
from werkzeug.utils import secure_filename
import tempfile
import io
import alpr_api
import concurrent.futures
import sqlite3
import random
import string
import argparse
import traceback
import glob
import logging
import json
import os
import sys
# Allows importing from: ../../
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("In [{0}], appending [{1}] to system path.".format(__file__, parent))
sys.path.append(parent)


UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logging.basicConfig(filename='alpr.log', level=logging.INFO)

TPE = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def auth_check(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "login_id" not in session:
            logging.warning("Illegal access to operation. Login required.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper


def get_ts_str():
    return DT.now().strftime("%Y%m%d_%H%M%S")


def random_str(size=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=size))


def _init_db():
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS users
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            login_id text NOT NULL UNIQUE, 
            pass_hashed text NOT NULL, full_name text NOT NULL, 
            role text NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS frames
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            client_id text NOT NULL,
            frame_file text NOT NULL UNIQUE, 
            received_ts text NOT NULL,
            status text NOT NULL, 
            task_type text NOT NULL,
            detected_text text)''')
        conn.commit()
        logging.info("DB initialized.")


def _authenticate(login_id, plain_pass):
    valid = False
    try:
        with sqlite3.connect('app.db') as conn:
            c = conn.cursor()
            # Create table
            c.execute(
                'SELECT pass_hashed FROM users WHERE login_id=?', (login_id,))
            row = c.fetchone()
            if row:
                valid = pbkdf2_sha256.verify(plain_pass, row[0])
    except Exception as ex:
        logging.exception("Error occurred when authenticating.")
    return valid


def _add_user(login_id, pass_hashed, full_name, role="USER"):
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute('SELECT count(*) FROM users WHERE login_id=?', (login_id,))
        if c.fetchone()[0] != 0:
            raise Exception("Login ID already exists.")
        c.execute("""INSERT INTO users(login_id, pass_hashed, full_name, role)
        VALUES (?,?,?,?)""", (login_id, pass_hashed, full_name, role))
        conn.commit()


def _add_frame(client_id, frame_file, task_type):
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO frames(client_id, 
        frame_file, received_ts, status, task_type)
        VALUES (?,?,?,?,?)""",
                  (client_id, frame_file, get_ts_str(), "NEW", task_type))
        conn.commit()


def _update_frame(frame_file, status, detected_text):
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute("""UPDATE frames SET status=?, detected_text=?
        WHERE frame_file=?""",
                  (status, detected_text, frame_file))
        conn.commit()
        return conn.total_changes


def _get_frames(client_id):
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute("""SELECT id, received_ts, status, 
        detected_text, frame_file, task_type 
        FROM frames WHERE client_id=? 
        order by id desc""",
                  (client_id,))
        return c.fetchall()

def _get_frame(id):
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute("""SELECT id, client_id, frame_file, 
        received_ts, status, detected_text, task_type 
        FROM frames WHERE id=?""", (id,))
        return c.fetchone()


def _fmt_date_str(dt_str):
    try:
        d2 = DT.strptime(dt_str, "%Y%m%d_%H%M%S")
        return d2.strftime("%Y-%b-%d@%H:%M:%S")
    except Exception as ex:
        logging.exception("Failed to format date.")
        return dt_str


@app.route('/signup', methods=['POST', 'GET'])
def signup():
    error = None
    try:
        if request.method == 'POST':
            pw_hashed = pbkdf2_sha256.hash(request.form['password'])
            _add_user(request.form['login_id'], pw_hashed,
                      request.form['full_name'])
            return render_template("index.html",
                                   error="User created. Please login with your credentials.")

    except Exception as ex:
        logging.exception("Error occurred when signing up.")
        error = str(ex)

    return render_template('signup.html', error=error)


@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    try:
        if request.method == 'POST':
            if _authenticate(request.form['login_id'],
                             request.form['password']):
                logging.info("Login successful.")
                session['login_id'] = request.form['login_id']
                return redirect(url_for('home'))
            else:
                error = 'Invalid username/password'
    except Exception as ex:
        logging.exception("Error occurred when logging in.")
        error = str(ex)
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('index.html', error=error)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('login_id', None)
    return redirect(url_for('index'))


def _to_bw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    contrast = clahe.apply(blurred)
    ret, bw_img = cv2.threshold(
        contrast, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    WHITE = [255, 255, 255]
    # Extend the border by 10px
    final_img = cv2.copyMakeBorder(
        bw_img.copy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=WHITE)
    return final_img


def _extract_boxes(img, boxes, out_dir):
    idx = 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for box_rec in boxes["box_info"]:
        print("BOX IS: "+str(box_rec))
        try:
            if box_rec["score"] < 0.2:
                logging.info(
                    "Skipping box with score {0}.".format(box_rec["score"]))
                continue
            bx = box_rec["box"]
            pad = 10
            logging.info("Padding by {0}".format(pad))
            roi = img[bx[0][1] - pad : bx[2][1] + pad, bx[0][0] - pad : bx[2][0] + pad]
            #roi = img[int(line["y0"]):int(line["y2"]), int(line["x0"]):int(line["x2"])]
            out_img = os.path.join(out_dir, "box_{0}.jpg".format(idx))
            #cv2.imwrite(out_img, roi)
            cv2.imwrite(out_img, _to_bw(roi))
            idx += 1
        except Exception as ex:
            logging.exception("Failed to extract box. Continuing to next one.")

    return idx


def _process_image(task_type, file_path, login_id):
    logging.info("Processing file {0} for task {1}".format(file_path, task_type))
    try:
        task_dir = file_path[:-len("input_frame.jpg")]
        Path(task_dir).mkdir(parents=True, exist_ok=True)

        if task_type == "ocr":
            img = cv2.imdecode(np.fromfile(file_path, dtype='uint8'), 1)
            boxes = alpr_api.get_predictor(checkpoint_path)(img)
            box_count = _extract_boxes(img, boxes, task_dir)
            if box_count > 0:
                rc = _update_frame(
                    file_path, "EXT", "Extracted {} boxes.".format(box_count))
                if rc > 0:
                    _invoke_alpr_api(file_path, task_dir, task_type)
                else:
                    logging.error(
                        "Failed to update the frame status in DB for: {}".format(file_path))
            else:
                rc = _update_frame(file_path, "FIN", "No text box found.")
                logging.info("No text box found in file: {}".format(file_path))
                if rc < 1:
                    logging.error(
                        "Failed to close the status for file {}".format(file_path))
        elif task_type == "alpr":
            _invoke_alpr_api(file_path, task_dir, task_type)
        else:
            logging.error("Invalid task type supplied: {}".format(task_type))

    except Exception as ex:
        logging.exception("Error occurred when processing image {}".format(file_path))

def _invoke_alpr_api(file_path, out_dir, task_type):
    logging.info("Starting task {0} in file: {1}".format(task_type, file_path))
    data = {"task_dir": out_dir, "task_type": task_type}
    r = requests.post(CONFIG["alpr_api_url"], json=data)
    if r.status_code != requests.codes.ok:
        logging.error("API service failed to process request. "+r.text)
    else:
        logging.info("API service result: {0}".format(r.text))


@app.route('/images/<int:id>')
@auth_check
def base_static(id):
    row = _get_frame(id)
    if row:
        return send_file(row[2])
    else:
        logging.error("No row found for ID {}".format(id))

@app.route('/home', methods=['GET', 'POST'])
@auth_check
def home():
    login_id = session['login_id']
    logging.info("Upload destination: "+UPLOAD_FOLDER)
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'img_file' not in request.files or "task_type" not in request.form:
                msg = "Required arguments not found in request."
                logging.info(msg)
                return render_template('home.html',
                                       error=msg,
                                       name=escape(login_id))
            file = request.files['img_file']
            task_type = request.form['task_type']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return render_template('home.html',
                                       error="No file data found!",
                                       name=escape(login_id))
            ext = os.path.splitext(file.filename)[1]
            print("File extension: "+ext)
            if file and ext in [".jpg", ".png", ".jpeg"]:
                sfn = secure_filename(file.filename)
                task_dir = os.path.join(app.config['UPLOAD_FOLDER'],
                                        login_id, get_ts_str())
                file_path = os.path.join(task_dir, "input_frame.jpg")
                Path(task_dir).mkdir(parents=True, exist_ok=True)
                file.save(file_path)
                logging.info("Saved the uploaded file to {0}".format(file_path))
                _add_frame(login_id, file_path, task_type)
                TPE.submit(_process_image, task_type, file_path, login_id)
                return redirect(url_for('show_status'))
            else:
                logging.error("File type {0} not allowed!".format(ext))
                return render_template('home.html',
                                       error="File type not allowed!",
                                       name=escape(login_id))

        else:
            logging.info("GET request for upload.")

        return render_template('home.html',
                               name=escape(login_id))
    except Exception as ex:
        logging.exception("Error when uploading.")
        return render_template('home.html', error=str(ex),
                               name=escape(login_id))


@app.route('/status', methods=['GET'])
@auth_check
def show_status():
    login_id = session['login_id']
    try:
        frames = _get_frames(login_id)
        logging.info("Found {0} frames".format(len(frames)))
        return render_template('status.html', data=frames,
                               name=escape(login_id))
    except Exception as ex:
        return render_template('status.html', error=str(ex),
                               name=escape(login_id))


CONFIG = None
checkpoint_path = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", type=bool, nargs='?',
                        const=True, default=False,
                        dest="debug", help="Run the server in debug mode.")
    parser.add_argument("cfg_file_path", type=str,
                        help="Scrapper runner config file path.")
    args = parser.parse_args()
    app.secret_key = random_str(size=30)
    _init_db()

    with open(args.cfg_file_path, "r") as cfg_file:
        CONFIG = json.load(cfg_file)

    logging.info("CONFIG: "+str(CONFIG))
    checkpoint_path = CONFIG["checkpoint_path"]
    app.run(host=CONFIG["host"],
            port=CONFIG["port"],
            threaded=True,
            ssl_context=(CONFIG["ssl_cert_file"], CONFIG["ssl_key_file"]),
            debug=args.debug)
