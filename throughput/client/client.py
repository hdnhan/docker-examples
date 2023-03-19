from pathlib import Path
import sys
import grpc
import time
import numpy as np

ROOT = Path(__file__).parent
sys.path.append((ROOT / "../proto").as_posix())

import tunnel_pb2
import tunnel_pb2_grpc


grpc_options = (
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
)

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50052", grpc_options)
    stub = tunnel_pb2_grpc.BenchServiceStub(channel)

    ntimes = 500
    exec_times = []
    h, w = 512, 512
    for i in range(ntimes):
        im = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        request = tunnel_pb2.Request(
            data=im.tobytes(),
            height=h,
            width=w,
        )
        start = time.perf_counter()
        response = stub.doInference(request)
        end = time.perf_counter()
        exec_times.append(end - start)

        # print(response.prediction, response.probability)
    print("Execution time: ", np.mean(exec_times), np.std(exec_times))
