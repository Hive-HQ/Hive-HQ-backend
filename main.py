import sys

sys.path.insert(0, "./yolov5")

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import urllib

server = Flask(__name__)
CORS(server)
ai_workers = {}
floorplan_image = None
heatmap_img = None
total_people = 0
total_dangerous = 0

# Hack way to allow multiple threads to talk to eachother lol
# GIL for the win!!!
class AIControlObj:
    def __init__(self, id):
        self.id = id
        self.final_img = None  # Final Output Frame
        self.final_plot_img = None  # Final Plot Frame
        self.online = True

        self.points = []

        self.p_x1 = 0
        self.p_y1 = 0
        self.p_x2 = 640
        self.p_y2 = 0
        self.p_x3 = 640
        self.p_y3 = 480
        self.p_x4 = 0
        self.p_y4 = 480

        self.o_x1 = 0
        self.o_y1 = 0
        self.o_x2 = 640
        self.o_y2 = 0
        self.o_x3 = 640
        self.o_y3 = 480
        self.o_x4 = 0
        self.o_y4 = 480


def start_detection(source, control_obj, out="inference/output", device="0"):
    yolo_weights = "yolov5/weights/yolov5l.pt"
    deepsort_config = "deep_sort_pytorch/configs/deep_sort.yaml"
    img_size = 640

    # p_x1 = 40
    # p_y1 = 130
    # p_x2 = 150
    # p_y2 = 120
    # p_x3 = 470
    # p_y3 = 350
    # p_x4 = -100
    # p_y4 = 590

    # o_x1 = 100
    # o_y1 = 100
    # o_x2 = 300
    # o_y2 = 100
    # o_x3 = 300
    # o_y3 = 400
    # o_x4 = 100
    # o_y4 = 400

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(deepsort_config)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )

    # Initialize
    device = select_device(device)

    half = device.type != "cpu"  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, img_size, img_size)
            .to(device)
            .type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        # print("Frame", frame_idx)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=[0])
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], "%g: " % i, im0s[i].copy()

            s += "%gx%g " % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            plot_width = 640
            plot_height = 480
            if floorplan_image is None:
                plotter = np.zeros((plot_height, plot_width, 3), np.uint8)
                plotter[:, 0:plot_width] = (250, 255, 255)
                cv2.putText(plotter, "Please Upload A Floorplan!", (20, 480-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
            else:
                plotter = cv2.resize(floorplan_image, dsize=(plot_width, plot_height), interpolation=cv2.INTER_CUBIC)

            cv2.polylines(
                annotator.im,
                [
                    np.array(
                        [
                            [
                                [control_obj.p_x1, control_obj.p_y1],
                                [control_obj.p_x2, control_obj.p_y2],
                                [control_obj.p_x3, control_obj.p_y3],
                                [control_obj.p_x4, control_obj.p_y4],
                            ]
                        ]
                    )
                ],
                True,
                (0, 255, 0),
                thickness=2,
            )

            cv2.polylines(
                plotter,
                [
                    np.array(
                        [
                            [
                                [control_obj.o_x1, control_obj.o_y1],
                                [control_obj.o_x2, control_obj.o_y2],
                                [control_obj.o_x3, control_obj.o_y3],
                                [control_obj.o_x4, control_obj.o_y4],
                            ]
                        ]
                    )
                ],
                True,
                (255, 0, 0),
                thickness=2,
            )

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    src = np.array(
                        (
                            (control_obj.p_x1, control_obj.p_y1), 
                            (control_obj.p_x2, control_obj.p_y2), 
                            (control_obj.p_x3, control_obj.p_y3), 
                            (control_obj.p_x4, control_obj.p_y4)),
                        dtype=np.float32,
                    )
                    dest = np.array(
                        (
                            (control_obj.o_x1, control_obj.o_y1), 
                            (control_obj.o_x2, control_obj.o_y2), 
                            (control_obj.o_x3, control_obj.o_y3), 
                            (control_obj.o_x4, control_obj.o_y4)
                        ),
                        dtype=np.float32,
                    )
                    mtx = cv2.getPerspectiveTransform(src, dest)

                    coordinates = []
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        # print(bboxes)
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f"{id} {names[c]} {conf:.2f}"

                        box_left = output[0]
                        box_top = output[1]
                        box_right = output[2]
                        box_bottom = output[3]

                        box_w = box_right - box_left
                        box_h = box_bottom - box_top

                        center_point_x = (box_left + box_right) // 2
                        center_point_y = box_bottom

                        coordinates.append((center_point_x, center_point_y))

                        angle = 0
                        startAngle = 0
                        endAngle = 360

                        cv2.rectangle(
                            annotator.im,
                            (center_point_x + 5, center_point_y + 5),
                            (center_point_x - 5, center_point_y - 5),
                            color=(0, 255, 0),
                            thickness=2,
                        )

                        cv2.ellipse(
                            annotator.im,
                            (center_point_x, center_point_y),
                            (box_w * 2, (2 * 2 * box_w) // (3)),
                            0,
                            0,
                            360,
                            (0, 165, 255),
                            thickness=2,
                        )

                        # Plotter

                        cv2.rectangle(
                            plotter,
                            (center_point_x + 5, center_point_y + 5),
                            (center_point_x - 5, center_point_y - 5),
                            color=(0, 0, 0),
                            thickness=2,
                        )

                        cv2.ellipse(
                            plotter,
                            (center_point_x, center_point_x),
                            (10, 10),
                            0,
                            0,
                            360,
                            (0, 165, 255),
                            thickness=2,
                        )

                    # print(coordinates)

                    transformed_coordinates = cv2.perspectiveTransform(
                        np.array([coordinates], dtype=np.float32),
                        mtx,
                    ).tolist()

                    transformed_coordinates = [
                        [int(x), int(y)] for x, y in transformed_coordinates[0]
                    ]

                    # print(transformed_coordinates)

                    control_obj.points = transformed_coordinates

                    for transformed_x, transformed_y in transformed_coordinates:
                        cv2.rectangle(
                            plotter,
                            (transformed_x + 3, transformed_y + 3),
                            (transformed_x - 3, transformed_y - 3),
                            color=(255, 255, 0),
                            thickness=2,
                        )

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print("[CAMERA %d] %sDone Frame. (%.3fs)" % (control_obj.id, s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            control_obj.final_img = im0
            control_obj.final_plot_img = plotter
            """
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration
            """
            if not control_obj.online:
                print("WORKER THREAD KILLED. (%.3fs)" % (time.time() - t0))
                return None


def generate_heatmap():
    while True:
        try:
            time.sleep(0.05)
            plot_width = 640
            plot_height = 480
            if floorplan_image is None:
                plotter = np.zeros((plot_height, plot_width, 3), np.uint8)
                plotter[:, 0:plot_width] = (250, 255, 255)
                cv2.putText(plotter, "Please Upload A Floorplan!", (20, 480-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
            else:
                plotter = cv2.resize(floorplan_image, dsize=(plot_width, plot_height), interpolation=cv2.INTER_CUBIC)

            temp_total_people = 0
            temp_total_dangerous = 0
            scanned_coords = []
            for worker in ai_workers.values():
                if worker.online:
                    cv2.polylines(
                        plotter,
                        [
                            np.array(
                                [
                                    [
                                        [worker.o_x1, worker.o_y1],
                                        [worker.o_x2, worker.o_y2],
                                        [worker.o_x3, worker.o_y3],
                                        [worker.o_x4, worker.o_y4],
                                    ]
                                ]
                            )
                        ],
                        True,
                        (255, 0, 0),
                        thickness=2,
                    )

                    for px, py in worker.points:
                        temp_total_people += 1
                        cv2.rectangle(
                            plotter,
                            (px + 5, py + 5),
                            (px - 5, py - 5),
                            color=(0, 0, 0),
                            thickness=2,
                        )

                        cv2.ellipse(
                            plotter,
                            (px, py),
                            (25, 25),
                            0,
                            0,
                            360,
                            (0, 165, 255),
                            thickness=2,
                        )

            global heatmap_img
            global total_people
            heatmap_img = plotter
            total_people = temp_total_people

        except Exception as e:
            print(e)
            time.sleep(1)
                


@server.route("/", strict_slashes=False)
def index():
    return "Server Active"


@server.route("/add_camera", methods=["POST"], strict_slashes=False)
def add_camera():
    json_data = request.get_json(force=True)
    try:
        ip = str(json_data["ip"])
    except KeyError:
        return "KeyError", 400
    except ValueError:
        return "ValueError", 400

    print("DEBUG: Creating New Camera")

    # Create threads for new camera
    global ai_workers
    worker_id = len(ai_workers)
    new_control_obj = AIControlObj(worker_id)
    new_ai_thread = threading.Thread(
        target=start_detection,
        args=(
            ip,
            new_control_obj,
        ),
    )
    new_ai_thread.daemon = True
    new_ai_thread.start()

    ai_workers[worker_id] = new_control_obj

    return str(worker_id)


@server.route("/get_cameras", strict_slashes=False)
def get_cameras():
    return jsonify([i for i, v in ai_workers.items() if v.online is True])


@server.route("/camera/<worker_id>/video.mjpg", strict_slashes=False)
def video_feed(worker_id):
    worker_id = int(worker_id)
    if worker_id not in ai_workers:
        return "Camera Not Found", 400
    elif ai_workers[worker_id] is None:
        return "Camera Offline", 400

    def generate_next_frame(control_obj):
        while True:
            time.sleep(0.1)
            ret, jpg = cv2.imencode(
                ".jpg", control_obj.final_img
            )
            frame = jpg.tobytes()
            yield (b"--HTN\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(
        generate_next_frame(ai_workers[worker_id]),
        mimetype="multipart/x-mixed-replace; boundary=HTN",
    )


@server.route("/camera/<worker_id>/plot.mjpg", strict_slashes=False)
def plot_feed(worker_id):
    worker_id = int(worker_id)
    if worker_id not in ai_workers:
        return "Camera Not Found", 400
    elif ai_workers[worker_id] is None:
        return "Camera Offline", 400

    def generate_next_frame(control_obj):
        while True:
            time.sleep(0.1)
            ret, jpg = cv2.imencode(
                ".jpg", control_obj.final_plot_img
            )
            frame = jpg.tobytes()
            yield (b"--HTN\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(
        generate_next_frame(ai_workers[worker_id]),
        mimetype="multipart/x-mixed-replace; boundary=HTN",
    )


@server.route("/heatmap.mjpg", strict_slashes=False)
def heatmap_feed():
    def generate_next_frame():
        while True:
            time.sleep(0.1)
            ret, jpg = cv2.imencode(
                ".jpg", heatmap_img
            )
            frame = jpg.tobytes()
            yield (b"--HTN\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(
        generate_next_frame(),
        mimetype="multipart/x-mixed-replace; boundary=HTN",
    )


@server.route("/set_floorplan", methods=["POST"], strict_slashes=False)
def set_floorplan():
    json_data = request.get_json(force=True)
    try:
        img_url = str(json_data["img_url"])
    except KeyError:
        return "KeyError", 400
    except ValueError:
        return "ValueError", 400

    if ".jpg" not in img_url and ".jpeg" not in img_url:
        return "JPEG ONLY", 400

    print("DEBUG: Creating New Camera")

    # Set Image
    global floorplan_image
    resp = urllib.request.urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    floorplan_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return "ok"


@server.route("/camera/<worker_id>/set_in_bounds", methods=["POST"], strict_slashes=False)
def set_in_bounds(worker_id):
    json_data = request.get_json(force=True)
    try:
        p_x1 = int(json_data["p_x1"])
        p_y1 = int(json_data["p_y1"])
        p_x2 = int(json_data["p_x2"])
        p_y2 = int(json_data["p_y2"])
        p_x3 = int(json_data["p_x3"])
        p_y3 = int(json_data["p_y3"])
        p_x4 = int(json_data["p_x4"])
        p_y4 = int(json_data["p_y4"])
    except KeyError:
        return "Missing Arguments", 400

    worker_id = int(worker_id)
    if worker_id not in ai_workers.keys():
        return "Camera Not Found", 400
    elif ai_workers[worker_id] is None:
        return "Camera Offline", 400

    ai_workers[worker_id].p_x1 = p_x1
    ai_workers[worker_id].p_y1 = p_y1
    ai_workers[worker_id].p_x2 = p_x2
    ai_workers[worker_id].p_y2 = p_y2
    ai_workers[worker_id].p_x3 = p_x3
    ai_workers[worker_id].p_y3 = p_y3
    ai_workers[worker_id].p_x4 = p_x4
    ai_workers[worker_id].p_y4 = p_y4

    return "ok"


@server.route("/camera/<worker_id>/set_out_bounds", methods=["POST"], strict_slashes=False)
def set_out_bounds(worker_id):
    json_data = request.get_json(force=True)
    try:
        o_x1 = int(json_data["o_x1"])
        o_y1 = int(json_data["o_y1"])
        o_x2 = int(json_data["o_x2"])
        o_y2 = int(json_data["o_y2"])
        o_x3 = int(json_data["o_x3"])
        o_y3 = int(json_data["o_y3"])
        o_x4 = int(json_data["o_x4"])
        o_y4 = int(json_data["o_y4"])
    except KeyError:
        return "Missing Arguments", 400

    worker_id = int(worker_id)
    if worker_id not in ai_workers.keys():
        return "Camera Not Found", 400
    elif ai_workers[worker_id] is None:
        return "Camera Offline", 400

    ai_workers[worker_id].o_x1 = o_x1
    ai_workers[worker_id].o_y1 = o_y1
    ai_workers[worker_id].o_x2 = o_x2
    ai_workers[worker_id].o_y2 = o_y2
    ai_workers[worker_id].o_x3 = o_x3
    ai_workers[worker_id].o_y3 = o_y3
    ai_workers[worker_id].o_x4 = o_x4
    ai_workers[worker_id].o_y4 = o_y4

    return "ok"


@server.route("/camera/<worker_id>/get_bounds", strict_slashes=False)
def get_bounds(worker_id):
    worker_id = int(worker_id)
    if worker_id not in ai_workers.keys():
        return "Camera Not Found", 400
    elif ai_workers[worker_id] is None:
        return "Camera Offline", 400

    return jsonify({
        "in": {
            "p_x1": ai_workers[worker_id].p_x1,
            "p_y1": ai_workers[worker_id].p_y1,
            "p_x2": ai_workers[worker_id].p_x2,
            "p_y2": ai_workers[worker_id].p_y2,
            "p_x3": ai_workers[worker_id].p_x3,
            "p_y3": ai_workers[worker_id].p_y3,
            "p_x4": ai_workers[worker_id].p_x4,
            "p_y4": ai_workers[worker_id].p_y4
        },
        "out": {
            "o_x1": ai_workers[worker_id].o_x1,
            "o_y1": ai_workers[worker_id].o_y1,
            "o_x2": ai_workers[worker_id].o_x2,
            "o_y2": ai_workers[worker_id].o_y2,
            "o_x3": ai_workers[worker_id].o_x3,
            "o_y3": ai_workers[worker_id].o_y3,
            "o_x4": ai_workers[worker_id].o_x4,
            "o_y4": ai_workers[worker_id].o_y4
        }
    })


@server.route("/camera/<worker_id>/stop", strict_slashes=False)
def stop_worker(worker_id):
    worker_id = int(worker_id)
    if worker_id not in ai_workers.keys():
        return "Camera Not Found", 400
    elif ai_workers[worker_id] is None:
        return "Camera Offline", 400

    ai_workers[worker_id].online = False

    return "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--source", type=str, default="0", help="source")
    parser.add_argument("--cuda", action="store_true", help="cuda")
    parser.add_argument("--device", default=None, help="cuda device")
    parser.add_argument("--ip", default="0.0.0.0", help="server ip")
    parser.add_argument("--port", default="5500", help="server port")
    args = parser.parse_args()

    # Start Heatmap Generator
    heatmap_generation_thread = threading.Thread(target=generate_heatmap)
    heatmap_generation_thread.setDaemon(True)
    heatmap_generation_thread.start()

    server.run(host=args.ip, port=args.port)

    # server_thread = threading.Thread(target=start_flask_server)
    # server_thread.setDaemon(True)
    # server_thread.start()

    # with torch.no_grad():
    #     detect(
    #         args.source,
    #         "inference/output",
    #         args.device if args.device else "0" if args.cuda else "cpu",
    #     )
