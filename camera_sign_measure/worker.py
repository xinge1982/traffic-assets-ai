import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional
import urllib.parse
import pika
import json
import base64
import time
import os
import requests

from main import (
    initialize_pipeline,
    run_sign_pipeline
)


DEFAULT_CAMERA_CALIBRATION = {
    "image_width": 3840,
    "image_height": 2880,
    "camera_matrix": [
        [2630.828844872921, 0.0, 1905.969826707353],
        [0.0, 2630.828844872921, 1415.7948909681757],
        [0.0, 0.0, 1.0],
    ],
    "distortion_coefficients": [
        0.06288643729140438,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
}

print("Initializing sign pipeline...")

initialize_pipeline()

print("Sign pipeline ready.")

def load_image_base64(payload):

    image_base64 = payload.get(
        "image_base64"
    )

    image_url = payload.get(
        "image_url"
    )


    if image_base64:

        return image_base64


    if image_url:

        if not urllib.parse.urlparse(image_url).scheme in (
            "http",
            "https"
        ):
            raise Exception(
                "只支持HTTP/HTTPS协议的image_url"
            )


        response = requests.get(
            image_url,
            timeout=30
        )


        if response.status_code != 200:
            raise Exception(
                f"下载失败: {response.status_code}"
            )


        encoded = base64.b64encode(
            response.content
        ).decode(
            "utf-8"
        )


        return (
            "data:image/jpeg;base64,"
            + encoded
        )


    raise Exception(
        "missing image_base64 or image_url"
    )

def get_camera_calibration(payload):
    """
    Read camera calibration parameters from payload.camera_calibration.
    Use the built-in calibration when the field is absent or null.
    """
    camera_calibration = payload.get(
        "camera_calibration"
    )

    if camera_calibration is None:
        print(
            "     camera_calibration not provided; "
            "using default calibration."
        )
        return DEFAULT_CAMERA_CALIBRATION

    if not isinstance(camera_calibration, dict):
        raise ValueError(
            "payload.camera_calibration must be a JSON object"
        )

    return camera_calibration


# 获取环境变量的值，如果变量不存在则返回默认值
def get_config(key, default=None):
    return os.getenv(key, default)

# 响应模型（保持原有代码不变）
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

def on_request(ch, method, properties, body):
    output_path = ""
    response_data = {}
    task_id = ""
    user_id = ""
    try:
        # 解析消息体
        # {
        #   "payload": {
        #     "action": "sign_measure",
        #     "image_url": "https://example.com/image.jpg",
        #     "camera_calibration": {
        #       "image_width": 3840,
        #       "image_height": 2880,
        #       "camera_matrix": [
        #         [2630.828844872921, 0.0, 1905.969826707353],
        #         [0.0, 2630.828844872921, 1415.7948909681757],
        #         [0.0, 0.0, 1.0]
        #       ],
        #       "distortion_coefficients": [
        #         0.06288643729140438,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0
        #       ]
        #     }
        #   }
        # }

        data = json.loads(body)
        task_id = data["task_id"]
        user_id = data["user_id"]
        payload = data["payload"]

        # 处理任务
        print(f" [✔] Received Task: {task_id}, User: {user_id}")
        print(f"     Payload: {payload['action']}")

        # 提取 Base64 图片数据
        image_base64 = payload.get("image_base64")
        image_url = payload.get("image_url")

        # 处理文件上传
        if image_base64:
            img_bytes = base64.b64decode(image_base64)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 处理URL
        elif image_url:
            if not urllib.parse.urlparse(image_url).scheme in ("http", "https"):
                raise Exception("只支持HTTP/HTTPS协议的image_url")

            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                # 将响应内容转换为 NumPy 数组
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                # 使用 OpenCV 读取图片
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                raise Exception(f"下载失败，状态码: {response.status_code}")

        if img is None:
            raise Exception("无法解码图片数据")

        image_base64 = load_image_base64(
            payload
        )

        camera_calibration = get_camera_calibration(
            payload
        )

        result = run_sign_pipeline(
            image_base64,
            camera_calibration
        )

        print(f" [✔] Task {task_id}: Image detect done ")

        # 任务处理
        response_data = {
            "task_id": task_id,
            "user_id": user_id,
            "payload": {
                "action": payload['action'],
                "result": result
            }
        }

    except (Exception, HTTPException) as e:
        print(f" [❌] Error processing message: {e}")
        # 发送响应到回调队列
        response_data = {
            "task_id": task_id,
            "user_id": user_id,
            "err": f"{e}",
        }
    finally:
        # 发送响应到回调队列
        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response_data)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)  # 确认消息已处理
def connect():
    """创建 RabbitMQ 连接"""
    while True:
        try:
            # 通过环境变量读取参数
            rabbitmq_host = get_config('RABBITMQ_HOST', '47.104.194.94')
            rabbitmq_port = get_config('RABBITMQ_PORT', 28197)
            rabbitmq_user = get_config('RABBITMQ_USER', 'rsu')
            rabbitmq_password = get_config('RABBITMQ_PASSWORD', 'Mapabc@2016')
            rabbitmq_vhost = get_config('RABBITMQ_VHOST', '/')
            print(f"connecting to rabbitmq: {rabbitmq_host}:{rabbitmq_port}")

            # 创建连接
            credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=rabbitmq_host,
                port=rabbitmq_port,
                virtual_host=rabbitmq_vhost,
                credentials=credentials,
                heartbeat=300,  # 心跳时间（秒）
                blocked_connection_timeout=300  # 阻塞超时（秒）
            )

            # 建立连接
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            print(f' [✔] rabbitmq connected: {rabbitmq_host}:{rabbitmq_port}')

            return connection, channel
        except pika.exceptions.AMQPError as e:
            print(f"❌ 连接失败，10 秒后重试... 错误: {e}")
            time.sleep(10)

while True:
    try:
        connection, channel = connect()

        # 声明队列
        channel.queue_declare(queue='yolo_model_queue')
        print(f"     rabbitmq queue: yolo_model_queue")

        # 设置预取值，让每个 worker 一次最多处理 2 条消息
        channel.basic_qos(prefetch_count=5)

        channel.basic_consume(queue='yolo_model_queue', on_message_callback=on_request)

        print("     waiting job...")
        channel.start_consuming()

    except KeyboardInterrupt:
        print("\nexit.")

    except (pika.exceptions.AMQPError, pika.exceptions.AMQPConnectionError, pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed) as e:
        print(f"⚠️ RabbitMQ 连接丢失，尝试重连... 错误: {e}")
        time.sleep(5)

    except Exception as e:
        print(f"🚨 未知错误: {e}")
        time.sleep(5)