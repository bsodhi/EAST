import os

import time
import datetime
import cv2
import numpy as np
import json

import functools
import logging
import collections
from imutils import contours
from pathlib import Path
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf
    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(
        ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'box_info': [
                {
                    'score': float,
                    'box': [[x0, y0],..[x3, y3]]
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net':,
                'restore':,
                'nms':
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            (boxes, boundingBoxes) = contours.sort_contours(
                boxes, method="top-to-bottom")

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        box_info = []
        if boxes is not None:
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                box_info.append({"box": box, "score": float(score)})
        else:
            logger.info("***** No boxes found!")
        ret = {
            'box_info': box_info,
            'rtparams': rtparams,
            'timing': timer,
        }
        return ret

    return predictor


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


def extract_text_boxes(file_path, task_dir, checkpoint_path):
    
    img = cv2.imdecode(np.fromfile(file_path, dtype='uint8'), 1)
    boxes = get_predictor(checkpoint_path)(img)
    box_count = 0
    for box_rec in boxes["box_info"]:
        try:
            if box_rec["score"] < 0.2:
                logging.info(
                    "Skipping box with score {0}.".format(box_rec["score"]))
                continue
            bx = box_rec["box"]
            pad = 10
            logging.info("Padding by {0}".format(pad))
            roi = img[bx[0][1] - pad: bx[2][1] +
                      pad, bx[0][0] - pad: bx[2][0] + pad]
            #roi = img[int(line["y0"]):int(line["y2"]), int(line["x0"]):int(line["x2"])]
            out_img = os.path.join(task_dir, "box_{0}.jpg".format(box_count))
            #cv2.imwrite(out_img, roi)
            cv2.imwrite(out_img, _to_bw(roi))
            box_count += 1
        except Exception as ex:
            logging.exception("Failed to extract box. Continuing to next one.")

    return box_count
