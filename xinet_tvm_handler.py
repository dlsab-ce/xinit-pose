"""
Video Anonymization Service using face detection and blurring.

This microservice provides HTTP endpoints for anonymizing images by detecting
and blurring faces (and optionally license plates). It integrates with the
digitalhub-servicegraph framework for real-time video stream processing.
"""
import json
import numpy as np
import cv2
import matplotlib.cm as cm
import onnxruntime as ort
import yolo_nms
import nuclio_sdk
import urllib.request

IMG_SZ=(224,224)

sk = [15,13, 13,11, 16,14, 14,12, 11,12, 
            5,11, 6,12, 5,6, 5,7, 6,8, 7,9, 8,10, 
            1,2, 0,1, 0,2, 1,3, 2,4, 3,5, 4,6]


def init_context(context:nuclio_sdk.Context, tvm_func:str):
    setattr(context, "tvm_func", tvm_func)
    context.logger.info(f"XiNet pose model url: {tvm_func}")


def preprocess_img(frame):
    # breakpoint()
    img = frame[:, :, ::-1]
    img = img/255.00
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def model_inference(context, input=None):
    tvm_func_url = getattr(context, 'tvm_func', None)
    if tvm_func_url is None:
        raise ValueError("Model url not valid")
    # call model signature
    meta = json.load(urllib.request.urlopen(f"{tvm_func_url}"))
    spec = meta["inputs"][0]
    _, _, height, width = spec["shape"] 
    # call the Open Inference v2 endpoint
    body = {"inputs": [{"name": spec["name"], "datatype": "FP32",
                        "shape": list(input.shape)}]}
    context.logger.info(f"body tvm request:{body}")
    body["inputs"][0]["data"] = input.ravel().tolist()
    request = urllib.request.Request(f"{tvm_func_url}/infer", data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
    tvm_url = urllib.request.urlopen(request, timeout=300)
    context.logger.info(f"Status Code: {tvm_url.getcode()}")
    result = json.load(tvm_url)
    output = result["outputs"][0]
    pred = np.asarray(output["data"], dtype=np.float32).reshape(output["shape"])    
    return pred


def post_process_multi(img, output, score_threshold=10):
    boxes, conf_scores, keypt_vectors = yolo_nms.non_max_suppression(output, score_threshold)

    for keypts, conf in zip(keypt_vectors, conf_scores):
        plot_keypoints(img, keypts, score_threshold)
    return img


def plot_keypoints(img, keypoints, threshold=10):
    for i in range(0,len(sk)//2):
        pos1 = (int(keypoints[3*sk[2*i]]), int(keypoints[3*sk[2*i]+1]))
        pos2 = (int(keypoints[3*sk[2*i+1]]), int(keypoints[3*sk[2*i+1]+1]))
        conf1 = keypoints[3*sk[2*i]+2]
        conf2 = keypoints[3*sk[2*i+1]+2]

        color = (cm.jet(i/(len(sk)//2))[:3])
        color = [int(c * 255) for c in color[::-1]]
        if conf1>threshold and conf2>threshold: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(img, pos1, pos2, color, thickness=8)

    for i in range(0,len(keypoints)//3):
        x = int(keypoints[3*i])
        y = int(keypoints[3*i+1])
        conf = keypoints[3*i+2]
        if conf > threshold: # Only draw the circle if confidence is above some threshold
            cv2.circle(img, (x, y), 3, (0,0,0), -1)


def handler(context, request):
    """
    Detect pose keypoints in the provided image.
    
    Expects: Image bytes in request body
    
    Returns: Image with detected keypoints
    """     
    try:
        # Get image from request body
        context.logger.info(f"request: {type(request)}")
        # check if request object has body attribute, if not use inputs
        if (hasattr(request, 'body') and request.body is not None):
            context.logger.info(f"request body: {type(request.body)} - {request.body.keys()}")
            data = request.body['inputs'][0]['data'][0] if request.body['inputs'][0]['datatype'] == "BYTES" else request.body['inputs'][0]['data']
        else:
            data = request.inputs[0].data[0] if request.inputs[0].datatype == "BYTES" else request.inputs[0].data
        image_bytes = bytes(data)
        context.logger.info(f"get image bytes {len(image_bytes)}")

        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        context.logger.info(f"array to process {len(image)}")
    
        if image is None:
            raise ValueError("Failed to decode image")
        
        image = cv2.resize(image, IMG_SZ)
        input_img = preprocess_img(image)
        output = model_inference(context, input_img)
        context.logger.info(f"get inference {output.shape}")
        
        # frame = post_process_single(frame, output[0], score_threshold=0.2)
        image = post_process_multi(image, output[0], score_threshold=0.2)
        context.logger.info(f"post process image {len(image)}")

        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', image)
        output_data = list(buffer.tobytes())
        context.logger.info(f"output_data {len(output_data)}")
        # Return anonymized image
        return context.Response(
            body=json.dumps({
                "outputs": [
                    {
                        "name": "image",
                        "datatype": "UINT8",
                        "shape": [1, len(output_data)],
                        "data": output_data
                    }
                ]
            }),
            headers={},
            content_type="application/json",
            status_code=200
        )
        
    except Exception as e:
        context.logger.error(f"Error processing image: {e}")
        return context.Response(
            body=json.dumps({
                "error": str(e),
                "status": "error"
            }),
            content_type="application/json",
            status_code=500
        )



