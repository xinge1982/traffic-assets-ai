#!/usr/bin/env python3
"""RabbitMQ single-image CLIP classification worker (Python 3.11+).

Place this file in traffic-assets-ai/sign_classification/.
Install from the repository root:
  python -m pip install -r sign_classification/requirements.txt
  python -m pip install "pika>=1.3,<2" "requests>=2.32,<3"

Run from the repository root:
  python sign_classification/sign_classification_worker.py
Local one-image check (no RabbitMQ connection):
  python sign_classification/sign_classification_worker.py --image D:/signs/test.png

Environment:
  SIGN_MODEL_DIR       default: output/ beside this script; contains classifier.joblib
  CLIP_DEVICE          auto (default), cpu, cuda, cuda:0
  CLIP_PRETRAINED      optional identical local CLIP weights; otherwise use training config
  RABBITMQ_HOST        localhost
  RABBITMQ_PORT        5672
  RABBITMQ_USER        required in queue mode
  RABBITMQ_PASSWORD    required in queue mode
  RABBITMQ_VHOST       /
  RABBITMQ_QUEUE       sign_classification_queue
  RABBITMQ_QUEUE_DURABLE false (must match existing queue)
  RABBITMQ_HEARTBEAT   60 seconds
  MAX_IMAGE_BYTES     20971520 (20 MiB)
  MAX_IMAGE_PIXELS    25000000

Request body (publish with reply_to and correlation_id):
  {"task_id":"task-001","user_id":"u1","payload":{
    "action":"sign_classification","image_url":"https://host/sign.png",
    "top_k":3,"min_probability":0.6}}
Use image_base64 instead of image_url for raw Base64 or data:image/...;base64,... .
If both are supplied, image_base64 takes precedence. One request = one sign image.

Success:
  {"task_id":"task-001","user_id":"u1","payload":{
    "action":"sign_classification","result":{"class_name":"...",
    "probability":0.95,"accepted":true,"top_k":[
    {"class_name":"...","probability":0.95}]}}}
Failure: {"task_id":"task-001","user_id":"u1","err":"..."}

CLIP and the trained head load once per process. This uses the Python training
artifacts directly, not ONNX. Input is a sign rendering/crop, not a full-road
object detector. Load only trusted classifier.joblib files.

Transport semantics: prefetch=1; inference runs on one background thread; all
publish/ack operations run on the connection thread. Publish confirms are enabled.
A job is acknowledged only after its result is published successfully. Connection
loss can cause redelivery: the caller should deduplicate replies using task_id.
Missing/expired reply queues cause rejection (dead-lettered if configured); an
error result is published and acknowledged for bad input/inference failures.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import signal
import threading
from urllib.parse import urlparse

from PIL import Image, ImageOps

LOG = logging.getLogger("sign-classification")
ACTION = "sign_classification"


def positive_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def read_image_bytes(payload: dict, max_bytes: int) -> bytes:
    """Download once, or decode raw/data-URI Base64. No temporary image files."""
    encoded = payload.get("image_base64")
    if encoded:
        if not isinstance(encoded, str):
            raise ValueError("image_base64 must be a string")
        # Bound whitespace too, before allocating a normalized copy.
        if len(encoded) > 4 * ((max_bytes + 2) // 3) + 4096:
            raise ValueError("image_base64 exceeds image size limit")
        if encoded.startswith("data:"):
            prefix, sep, encoded = encoded.partition(",")
            if not sep or not prefix.lower().startswith("data:image/") or ";base64" not in prefix.lower():
                raise ValueError("invalid Base64 image data URI")
        try:
            raw = base64.b64decode("".join(encoded.split()), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid image_base64") from exc
    else:
        url = payload.get("image_url")
        if not isinstance(url, str) or urlparse(url).scheme.lower() not in ("http", "https"):
            raise ValueError("provide image_base64 or an HTTP/HTTPS image_url")
        import requests

        with requests.get(url, stream=True, timeout=(5, 30)) as response:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("downloaded image exceeds size limit")
            chunks, count = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("downloaded image exceeds size limit")
                chunks.append(chunk)
            raw = b"".join(chunks)
    if not raw or len(raw) > max_bytes:
        raise ValueError("empty image or image exceeds size limit")
    return raw


def prepare_image(raw: bytes, max_pixels: int) -> Image.Image:
    """Same EXIF/white-alpha/square-padding policy as training common.load_image."""
    with Image.open(BytesIO(raw)) as source:
        if source.width * source.height > max_pixels:
            raise ValueError("image exceeds pixel limit")
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    side = max(rgb.size)
    if side * side > max_pixels:
        raise ValueError("square-padded image exceeds pixel limit")
    square = Image.new("RGB", (side, side), "white")
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return square


class SignClassifier:
    def __init__(self, model_dir: Path, device: str = "auto", pretrained: str | None = None):
        # Lazy imports let --help and transport tests run without CLIP downloads.
        import joblib
        import numpy as np
        import open_clip
        import torch

        artifact = joblib.load(model_dir / "classifier.joblib")
        config = artifact["config"]
        if config["format_version"] != 1 or config["preprocessing"] != "exif_alpha_white_square_pad_v1":
            raise ValueError("unsupported classifier format/preprocessing")
        self.head = artifact["classifier"]
        self.classes = config["classes"]
        indices = self.head.classes_.tolist()
        if (len(self.classes) < 2 or len(set(self.classes)) != len(self.classes)
                or any(not isinstance(i, int) for i in indices)
                or sorted(indices) != list(range(len(self.classes)))):
            raise ValueError("classifier classes do not match training configuration")
        self.feature_dim = int(config["feature_dim"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config["model"], pretrained=pretrained or config["pretrained"], device=self.device,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.torch, self.np = torch, np
        LOG.info("Model ready: %s, device=%s, classes=%d", config["model"], self.device, len(self.classes))

    def predict(self, image: Image.Image, top_k: int, min_probability: float) -> dict:
        torch, np = self.torch, self.np
        with torch.inference_mode():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(tensor).float()
            features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
        if features.shape != (1, self.feature_dim) or not np.isfinite(features).all():
            raise ValueError("invalid CLIP feature shape/values")
        probabilities = self.head.predict_proba(features)[0]
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1)):
            raise ValueError("invalid classifier probabilities")
        order = np.argsort(-probabilities, kind="stable")[:min(top_k, len(probabilities))]
        ranked = [{"class_name": self.classes[int(self.head.classes_[i])],
                   "probability": float(probabilities[i])} for i in order]
        return {"class_name": ranked[0]["class_name"], "probability": ranked[0]["probability"],
                "accepted": ranked[0]["probability"] >= min_probability, "top_k": ranked}


def process_request(body: bytes, classifier, max_bytes: int, max_pixels: int) -> dict:
    task_id, user_id = "", ""
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("message must be a JSON object")
        raw_task_id, raw_user_id = data.get("task_id", ""), data.get("user_id", "")
        if not isinstance(raw_task_id, (str, int)) or isinstance(raw_task_id, bool) or raw_task_id == "":
            raise ValueError("task_id must be a nonempty string or integer")
        task_id = raw_task_id
        if not isinstance(raw_user_id, (str, int)) or isinstance(raw_user_id, bool):
            raise ValueError("user_id must be a string or integer")
        user_id = raw_user_id
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        action = payload.get("action", ACTION)
        if action != ACTION:
            raise ValueError(f"unsupported action: {action!r}; expected {ACTION}")
        top_k = payload.get("top_k", 3)
        threshold = payload.get("min_probability", 0.0)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("min_probability must be a finite number in [0,1]")
        LOG.info("Received task=%s user=%s", task_id, user_id)
        image = prepare_image(read_image_bytes(payload, max_bytes), max_pixels)
        try:
            result = classifier.predict(image, top_k, float(threshold))
        finally:
            image.close()
        return {"task_id": task_id, "user_id": user_id,
                "payload": {"action": action, "result": result}}
    except Exception as exc:
        # No image payload, URL or credential logging. Client receives the reason.
        LOG.warning("Task failed: task=%s error_type=%s", task_id, type(exc).__name__)
        return {"task_id": task_id, "user_id": user_id, "err": str(exc)}


class QueueWorker:
    def __init__(self, classifier, max_bytes: int, max_pixels: int):
        import pika

        self.pika = pika
        self.classifier = classifier
        self.max_bytes, self.max_pixels = max_bytes, max_pixels
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sign-inference")
        self.stopping = threading.Event()
        self.connection = None
        self.channel = None

    def on_request(self, channel, method, properties, body):
        if not properties.reply_to:
            LOG.error("Rejecting task without reply_to: delivery=%s", method.delivery_tag)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        connection = self.connection
        future = self.executor.submit(process_request, body, self.classifier, self.max_bytes, self.max_pixels)

        def completed(done):
            try:
                response = done.result()
                connection.add_callback_threadsafe(partial(
                    self.publish_response, connection, channel, method.delivery_tag, properties, response,
                ))
            except self.pika.exceptions.ConnectionWrongStateError:
                LOG.warning("Connection closed before reply; unacknowledged task will be redelivered")
            except Exception:
                LOG.exception("Cannot schedule task reply")
        future.add_done_callback(completed)

    def publish_response(self, connection, channel, delivery_tag, properties, response):
        if not connection.is_open or not channel.is_open:
            return  # Never acknowledge an old delivery on a new channel.
        try:
            channel.basic_publish(
                exchange="", routing_key=properties.reply_to, mandatory=True,
                properties=self.pika.BasicProperties(
                    correlation_id=properties.correlation_id, content_type="application/json",
                    content_encoding="utf-8", delivery_mode=2,
                ),
                body=json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            )
        except self.pika.exceptions.UnroutableError:
            LOG.error("Reply queue unavailable; rejecting task=%s", response.get("task_id"))
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
            return
        except self.pika.exceptions.NackError:
            LOG.warning("Broker rejected reply; requeue task=%s", response.get("task_id"))
            channel.basic_nack(delivery_tag=delivery_tag, requeue=True)
            return
        channel.basic_ack(delivery_tag=delivery_tag)
        LOG.info("Replied task=%s success=%s", response.get("task_id"), "err" not in response)

    def stop(self, *_):
        self.stopping.set()
        connection, channel = self.connection, self.channel
        if connection and connection.is_open and channel and channel.is_open:
            try:
                connection.add_callback_threadsafe(channel.stop_consuming)
            except self.pika.exceptions.ConnectionWrongStateError:
                pass

    def run(self):
        pika = self.pika
        username, password = os.getenv("RABBITMQ_USER"), os.getenv("RABBITMQ_PASSWORD")
        if not username or not password:
            raise ValueError("set RABBITMQ_USER and RABBITMQ_PASSWORD")
        durable_value = os.getenv("RABBITMQ_QUEUE_DURABLE", "false").lower()
        if durable_value not in ("true", "false", "1", "0"):
            raise ValueError("RABBITMQ_QUEUE_DURABLE must be true/false/1/0")
        parameters = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "localhost"),
            port=positive_env("RABBITMQ_PORT", 5672),
            virtual_host=os.getenv("RABBITMQ_VHOST", "/"),
            credentials=pika.PlainCredentials(username, password),
            heartbeat=positive_env("RABBITMQ_HEARTBEAT", 60),
            blocked_connection_timeout=60, socket_timeout=10, stack_timeout=20,
        )
        queue = os.getenv("RABBITMQ_QUEUE", "sign_classification_queue")
        if not queue:
            raise ValueError("RABBITMQ_QUEUE cannot be empty")
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        try:
            while not self.stopping.is_set():
                try:
                    self.connection = pika.BlockingConnection(parameters)
                    self.channel = self.connection.channel()
                    self.channel.queue_declare(queue=queue, durable=durable_value in ("true", "1"))
                    self.channel.basic_qos(prefetch_count=1)
                    self.channel.confirm_delivery()
                    self.channel.basic_consume(queue=queue, on_message_callback=self.on_request, auto_ack=False)
                    LOG.info("Waiting for single-image tasks on queue=%s", queue)
                    self.channel.start_consuming()
                except pika.exceptions.ChannelClosedByBroker:
                    # E.g. mismatched queue durability: restarting cannot fix configuration.
                    LOG.error("Broker closed channel; check queue declaration/permissions")
                    raise
                except (pika.exceptions.AMQPError, OSError) as exc:
                    LOG.warning("RabbitMQ disconnected (%s); reconnecting", type(exc).__name__)
                finally:
                    if self.connection and self.connection.is_open:
                        try:
                            self.connection.close()
                        except pika.exceptions.AMQPError:
                            pass
                    self.connection, self.channel = None, None
                self.stopping.wait(5)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("SIGN_MODEL_DIR", str(Path(__file__).resolve().parent / "output"))))
    parser.add_argument("--device", default=os.getenv("CLIP_DEVICE", "auto"))
    parser.add_argument("--pretrained", default=os.getenv("CLIP_PRETRAINED"))
    parser.add_argument("--image", type=Path, help="local single-image check; does not connect to RabbitMQ")
    parser.add_argument("--top-k", type=int, default=3, help="for --image only")
    parser.add_argument("--min-probability", type=float, default=0.0, help="for --image only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    max_bytes = positive_env("MAX_IMAGE_BYTES", 20 * 1024 * 1024)
    max_pixels = positive_env("MAX_IMAGE_PIXELS", 25_000_000)
    if args.top_k < 1 or not math.isfinite(args.min_probability) or not 0 <= args.min_probability <= 1:
        parser.error("top-k must be positive; min-probability must be in [0,1]")
    classifier = SignClassifier(args.model_dir, args.device, args.pretrained)
    if args.image:
        # Reuse the same request and error path as queue processing.
        if args.image.stat().st_size > max_bytes:
            raise ValueError("local image exceeds size limit")
        request = {"task_id": "local-check", "user_id": "", "payload": {
            "action": ACTION, "image_base64": base64.b64encode(args.image.read_bytes()).decode("ascii"),
            "top_k": args.top_k, "min_probability": args.min_probability,
        }}
        response = process_request(json.dumps(request).encode(), classifier, max_bytes, max_pixels)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 1 if "err" in response else 0
    QueueWorker(classifier, max_bytes, max_pixels).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
