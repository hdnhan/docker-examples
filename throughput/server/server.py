import cv2
from pathlib import Path
import sys
import grpc
from concurrent import futures
import numpy as np
import time
import onnxruntime as ort

ROOT = Path(__file__).parent
sys.path.append((ROOT / "../proto").as_posix())

import tunnel_pb2
import tunnel_pb2_grpc


class OrtInference:
    def __init__(self) -> None:
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = False
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        self.sess = ort.InferenceSession(f"{ROOT}/model.onnx", sess_options)

        shape = self.sess.get_inputs()[0].shape
        self.h, self.w = shape[2], shape[3]

    def preprocess(self, image):
        transformed_roi = cv2.resize(image, (self.w, self.h))

        transformed_roi = transformed_roi.astype(np.float32)
        transformed_roi /= 255.0

        # Swap R <-> B
        transformed_roi[:, :, :3] = transformed_roi[:, :, ::-1]
        transformed_roi = (transformed_roi - self.mean) / self.std

        # Tranpose (H, W, 3) => (3, H, W)
        transformed_roi = transformed_roi.transpose(2, 0, 1)
        return transformed_roi

    def inference(self, image):
        input = {self.sess.get_inputs()[0].name: image[np.newaxis, :]}
        output = self.sess.run(None, input)[0][0]
        return output

    def postprocess(self, output):
        idx = np.argmax(output)
        return idx, np.exp(output[idx]) / np.sum(np.exp(output))


class BenchServiceServicer(tunnel_pb2_grpc.BenchServiceServicer):
    def __init__(self) -> None:
        self.sess = OrtInference()
        super().__init__()

    def doInference(self, request, context):
        np_data = np.frombuffer(request.data, dtype=np.uint8).reshape(
            request.height, request.width, 3
        )

        start = time.perf_counter()
        input = self.sess.preprocess(np_data)
        output = self.sess.inference(input)
        idx, val = self.sess.postprocess(output)
        end = time.perf_counter()

        print("Pre/Infer/Post(Py): ", end - start)
        response = tunnel_pb2.Response(prediction=idx, probability=val)
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tunnel_pb2_grpc.add_BenchServiceServicer_to_server(BenchServiceServicer(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    print("Server started on port 50052")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
