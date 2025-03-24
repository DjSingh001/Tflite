import tensorflow as tf
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=r"D:\YOLO_Test\Working_yolo_model\best_saved_model\best_float32.tflite")
interpreter.allocate_tensors()

# Retrieve input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

import numpy as np
from PIL import Image

# Load and resize the image
image = Image.open(r"D:\Frames\Minecraft\Frame_Minecraft_ [svCzpPRRj4M]_5447.jpg").resize((640, 640))
image_np = np.array(image)

# Normalize the image to [0, 1]
input_data = np.expand_dims(image_np / 255.0, axis=0).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Retrieve the output tensor
output = interpreter.get_tensor(output_details[0]['index'])[0]


x, y, w, h = output[0, :], output[1, :], output[2, :], output[3, :]

  

def iou(box1, box2):
    """Compute IoU (Intersection over Union) between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    # Compute intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes, confidences, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to remove overlapping boxes."""
    indices = np.argsort(confidences)[::-1]  # Sort by confidence (highest first)
    keep_boxes = []

    while len(indices) > 0:
        best_idx = indices[0]  # Box with highest confidence
        keep_boxes.append(best_idx)

        # Compare IoU with remaining boxes
        remaining_indices = indices[1:]
        new_indices = []

        for idx in remaining_indices:
            if iou(boxes[best_idx], boxes[idx]) < iou_threshold:
                new_indices.append(idx)

        indices = np.array(new_indices)

    return keep_boxes

# Load image
image_path = r"D:\Frames\Minecraft\Frame_Minecraft_ [svCzpPRRj4M]_5447.jpg"  # Replace with your image path
image = cv2.imread(image_path)
height, width, _ = image.shape  # Get image dimensions

# Extract YOLO output
output_array = output  # Shape (81, 8400)

# Extract x, y, w, h from the first 4 rows
x, y, w, h = output[0, :], output[1, :], output[2, :], output[3, :]

# Extract class probabilities (77 classes from row 4 to 80)
class_scores = output[4:81, :]  # Shape (77, 8400)

# Extract objectness scores from row 77

# Compute final confidence score
max_class_confidences = np.max(class_scores, axis=0)  # Max class confidence per detection
final_confidences = max_class_confidences  # Final confidence

# Find class IDs
class_ids = np.argmax(class_scores, axis=0)

# Filter by confidence threshold
conf_threshold = 0.6
valid_indices = np.where(final_confidences > conf_threshold)[0]

# Get filtered boxes and scores
filtered_boxes = np.array([x[valid_indices], y[valid_indices], w[valid_indices], h[valid_indices]]).T
filtered_confidences = final_confidences[valid_indices]
filtered_class_ids = class_ids[valid_indices]

# Apply NMS
nms_indices = non_max_suppression(filtered_boxes, filtered_confidences, iou_threshold=0.5)


import numpy as np
import cv2

def iou(box1, box2):
    """Compute IoU (Intersection over Union) between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    # Compute intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes, confidences, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to remove overlapping boxes."""
    indices = np.argsort(confidences)[::-1]  # Sort by confidence (highest first)
    keep_boxes = []

    while len(indices) > 0:
        best_idx = indices[0]  # Box with highest confidence
        keep_boxes.append(best_idx)

        # Compare IoU with remaining boxes
        remaining_indices = indices[1:]
        new_indices = []

        for idx in remaining_indices:
            if iou(boxes[best_idx], boxes[idx]) < iou_threshold:
                new_indices.append(idx)

        indices = np.array(new_indices)

    return keep_boxes

# Load image
image_path = r"D:\Frames\Minecraft\Frame_Minecraft_ [svCzpPRRj4M]_7427.jpg"  # Replace with your image path
image = cv2.imread(image_path)
height, width, _ = image.shape  # Get image dimensions

# Extract YOLO output
 # Shape (81, 8400)

# Extract x, y, w, h from the first 4 rows
x, y, w, h = output[0, :], output[1, :], output[2, :], output[3, :]

# Extract class probabilities (77 classes from row 4 to 80)
class_scores = output[4:81, :]  # Shape (77, 8400)

# Extract objectness scores from row 77


# Compute final confidence score
max_class_confidences = np.max(class_scores, axis=0)  # Max class confidence per detection
final_confidences = max_class_confidences  # Final confidence

# Find class IDs
class_ids = np.argmax(class_scores, axis=0)

# Filter by confidence threshold
conf_threshold = 0.6
valid_indices = np.where(final_confidences > conf_threshold)[0]

# Get filtered boxes and scores
filtered_boxes = np.array([x[valid_indices], y[valid_indices], w[valid_indices], h[valid_indices]]).T
filtered_confidences = final_confidences[valid_indices]
filtered_class_ids = class_ids[valid_indices]

# Apply NMS
nms_indices = non_max_suppression(filtered_boxes, filtered_confidences, iou_threshold=0.5)

# Draw bounding boxes
for i in nms_indices:
    x, y, w, h = filtered_boxes[i]
    class_id = filtered_class_ids[i]
    conf = filtered_confidences[i]

    # Convert (x, y, w, h) from normalized (0-1) to pixel coordinates
    x1 = int((x - w / 2) * width)
    y1 = int((y - h / 2) * height)
    x2 = int((x + w / 2) * width)
    y2 = int((y + h / 2) * height)
    print(conf ," ", class_id)

    # Draw rectangle
    # Draw rectangle
    color = (0, 255, 0)  # Green color
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Label text
    label = f"Class {class_id} ({conf:.2f})"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Show image with bounding boxes
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




import tensorflow as tf
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=r"D:\YOLO_Test\Working_yolo_model\best_saved_model\best_float32.tflite")
interpreter.allocate_tensors()

# Retrieve input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

import numpy as np
from PIL import Image

# Load and resize the image
image = Image.open(r"D:\Frames\Minecraft\Frame_Minecraft_ [svCzpPRRj4M]_5447.jpg").resize((640, 640))
image_np = np.array(image)

# Normalize the image to [0, 1]
input_data = np.expand_dims(image_np / 255.0, axis=0).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Retrieve the output tensor
output = interpreter.get_tensor(output_details[0]['index'])[0]


x, y, w, h = output[0, :], output[1, :], output[2, :], output[3, :]

  

def iou(box1, box2):
    """Compute IoU (Intersection over Union) between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    # Compute intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes, confidences, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to remove overlapping boxes."""
    indices = np.argsort(confidences)[::-1]  # Sort by confidence (highest first)
    keep_boxes = []

    while len(indices) > 0:
        best_idx = indices[0]  # Box with highest confidence
        keep_boxes.append(best_idx)

        # Compare IoU with remaining boxes
        remaining_indices = indices[1:]
        new_indices = []

        for idx in remaining_indices:
            if iou(boxes[best_idx], boxes[idx]) < iou_threshold:
                new_indices.append(idx)

        indices = np.array(new_indices)

    return keep_boxes

# Load image
image_path = r"D:\Frames\Minecraft\Frame_Minecraft_ [svCzpPRRj4M]_5447.jpg"  # Replace with your image path
image = cv2.imread(image_path)
height, width, _ = image.shape  # Get image dimensions

# Extract YOLO output
output_array = output  # Shape (81, 8400)

# Extract x, y, w, h from the first 4 rows
x, y, w, h = output[0, :], output[1, :], output[2, :], output[3, :]

# Extract class probabilities (77 classes from row 4 to 80)
class_scores = output[4:81, :]  # Shape (77, 8400)

# Extract objectness scores from row 77

# Compute final confidence score
max_class_confidences = np.max(class_scores, axis=0)  # Max class confidence per detection
final_confidences = max_class_confidences  # Final confidence

# Find class IDs
class_ids = np.argmax(class_scores, axis=0)

# Filter by confidence threshold
conf_threshold = 0.6
valid_indices = np.where(final_confidences > conf_threshold)[0]

# Get filtered boxes and scores
filtered_boxes = np.array([x[valid_indices], y[valid_indices], w[valid_indices], h[valid_indices]]).T
filtered_confidences = final_confidences[valid_indices]
filtered_class_ids = class_ids[valid_indices]

# Apply NMS
nms_indices = non_max_suppression(filtered_boxes, filtered_confidences, iou_threshold=0.5)


import numpy as np
import cv2


# Load image
image_path = r"D:\Frames\Minecraft\Frame_Minecraft_ [svCzpPRRj4M]_7427.jpg"  # Replace with your image path
image = cv2.imread(image_path)
height, width, _ = image.shape  # Get image dimensions

# Extract YOLO output
 # Shape (81, 8400)

# Extract x, y, w, h from the first 4 rows
x, y, w, h = output[0, :], output[1, :], output[2, :], output[3, :]

# Extract class probabilities (77 classes from row 4 to 80)
class_scores = output[4:81, :]  # Shape (77, 8400)

# Extract objectness scores from row 77


# Compute final confidence score
max_class_confidences = np.max(class_scores, axis=0)  # Max class confidence per detection
final_confidences = max_class_confidences  # Final confidence

# Find class IDs
class_ids = np.argmax(class_scores, axis=0)

# Filter by confidence threshold
conf_threshold = 0.6
valid_indices = np.where(final_confidences > conf_threshold)[0]

# Get filtered boxes and scores
filtered_boxes = np.array([x[valid_indices], y[valid_indices], w[valid_indices], h[valid_indices]]).T
filtered_confidences = final_confidences[valid_indices]
filtered_class_ids = class_ids[valid_indices]

# Apply NMS
nms_indices = non_max_suppression(filtered_boxes, filtered_confidences, iou_threshold=0.5)

# Draw bounding boxes
for i in nms_indices:
    x, y, w, h = filtered_boxes[i]
    class_id = filtered_class_ids[i]
    conf = filtered_confidences[i]

    # Convert (x, y, w, h) from normalized (0-1) to pixel coordinates
    x1 = int((x - w / 2) * width)
    y1 = int((y - h / 2) * height)
    x2 = int((x + w / 2) * width)
    y2 = int((y + h / 2) * height)
    print(conf ," ", class_id)

    # Draw rectangle
    # Draw rectangle
    color = (0, 255, 0)  # Green color
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Label text
    label = f"Class {class_id} ({conf:.2f})"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Show image with bounding boxes
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



















#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Function to perform Non-Maximum Suppression (NMS)
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, float iouThreshold) {
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0, iouThreshold, indices);
    return indices;
}

int main() {
    // Load the TFLite model
    const char* modelPath = "D:/YOLO_Test/Working_yolo_model/best_saved_model/best_float32.tflite";
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return -1;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return -1;
    }

    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return -1;
    }

    // Get input tensor information
    int inputIndex = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(inputIndex)->dims;
    int inputHeight = dims->data[1];
    int inputWidth = dims->data[2];
    int inputChannels = dims->data[3];

    // Load and preprocess the image
    cv::Mat image = cv::imread("D:/Frames/Minecraft/Frame_Minecraft_ [svCzpPRRj4M]_6635.jpg");
    if (image.empty()) {
        std::cerr << "Image not found\n";
        return -1;
    }
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(inputWidth, inputHeight));
    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0 / 255);

    // Set input tensor data
    float* inputTensor = interpreter->typed_tensor<float>(inputIndex);
    std::memcpy(inputTensor, resizedImage.data, resizedImage.total() * resizedImage.elemSize());

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite\n";
        return -1;
    }

    // Process output tensors
    int outputIndex = interpreter->outputs()[0];
    float* outputData = interpreter->typed_output_tensor<float>(outputIndex);

    // Assuming output tensor shape is [1, num_boxes, 7] with each box having [x, y, w, h, confidence, class_id]
    int numBoxes = interpreter->tensor(outputIndex)->dims->data[1];
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < numBoxes; ++i) {
        float x = outputData[i * 7 + 0];
        float y = outputData[i * 7 + 1];
        float w = outputData[i * 7 + 2];
        float h = outputData[i * 7 + 3];
        float confidence = outputData[i * 7 + 4];
        int classId = static_cast<int>(outputData[i * 7 + 5]);

        if (confidence > 0.6) { // Confidence threshold
            int left = static_cast<int>((x - w / 2) * image.cols);
            int top = static_cast<int>((y - h / 2) * image.rows);
            int width = static_cast<int>(w * image.cols);
            int height = static_cast<int>(h * image.rows);
            boxes.emplace_back(left, top, width, height);
            confidences.push_back(confidence);
            classIds.push_back(classId);
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nmsIndices = nonMaxSuppression(boxes, confidences, 0.5);

    // Draw bounding boxes
    for (int idx : nmsIndices) {
        cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(classIds[idx]) + " (" + std::to_string(confidences[idx]) + ")";
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(boxes[idx].y, labelSize.height);
        cv::putText(image, label, cv::Point(boxes[idx].x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Display the image
    cv::imshow("Detected Objects", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
















# lnet=len(output_data[0])
# print (lnet)

# # Define confidence threshold
# confidence_threshold = 0.5

# # Process each detection
# for detection in output_data:
#     x_center, y_center, width, height, confidence, *class_probs = detection
#     if confidence > confidence_threshold:
#         class_id = np.argmax(class_probs)
#         class_score = class_probs[class_id]
#         final_score = confidence * class_score
#         if final_score > confidence_threshold:
#             # Convert to corner coordinates
#             x_min = (x_center - width / 2) * image.width
#             y_min = (y_center - height / 2) * image.height
#             x_max = (x_center + width / 2) * image.width
#             y_max = (y_center + height / 2) * image.height
#             print(f"Detected object {class_id} with confidence {final_score:.2f} at [{x_min}, {y_min}, {x_max}, {y_max}]")

































# lnet=len(output_data[0])
# print (lnet)

# # Define confidence threshold
# confidence_threshold = 0.5

# # Process each detection
# for detection in output_data:
#     x_center, y_center, width, height, confidence, *class_probs = detection
#     if confidence > confidence_threshold:
#         class_id = np.argmax(class_probs)
#         class_score = class_probs[class_id]
#         final_score = confidence * class_score
#         if final_score > confidence_threshold:
#             # Convert to corner coordinates
#             x_min = (x_center - width / 2) * image.width
#             y_min = (y_center - height / 2) * image.height
#             x_max = (x_center + width / 2) * image.width
#             y_max = (y_center + height / 2) * image.height
#             print(f"Detected object {class_id} with confidence {final_score:.2f} at [{x_min}, {y_min}, {x_max}, {y_max}]")




import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from safetensors.torch import load_file
import plotly.express as px
from sklearn.manifold import TSNE
import psutil
import time
os.environ["TRANSFORMER_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# Load MobileViT model
mobilevit = timm.create_model(
    'mobilevitv2_075.cvnets_in1k',
    pretrained=False,
    num_classes=0  # Remove classifier
)

model_path = r"D:\Mobile_Vit\mobilevitv2_075.cvnets_in1k\mobilevitv2_075.cvnets_in1k\model.safetensors"  # Change to "pytorch_model.bin" if using bin file
if model_path.endswith(".safetensors"):
    state_dict = load_file(model_path)  # Load from safetensors
else:
    state_dict = torch.load(model_path, map_location="cpu")  # Load from bin file

# Remove classifier layer keys
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.fc")}
mobilevit.load_state_dict(filtered_state_dict, strict=False)  # Allow missing keys

mobilevit.eval()

# Define Projection Layer (384 → 512)
projection_layer = nn.Linear(384, 512).to("cpu")

# Load saved model
checkpoint = torch.load("distilled_mobilevit.pth")
mobilevit.load_state_dict(checkpoint["mobilevit_state_dict"])
projection_layer.load_state_dict(checkpoint["projection_layer_state_dict"])

# Set models to evaluation mode
mobilevit.eval()
projection_layer.eval()

# Define preprocessing transform (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image_path = r"D:\Mobile_Vit\embedding_dataset\black_myth_wukong_[0jFxQF4HPfM]_90055.jpg"

image = Image.open(image_path).convert("RGB")
print(type(image))
image_tensor = transform(image).unsqueeze(0).to("cpu")  # Shape: (1, 3, 224, 224)

# Extract embeddings using distilled MobileViT
with torch.no_grad():
    student_embeddings = mobilevit.forward_features(image_tensor)
    student_embeddings = mobilevit.forward_head(student_embeddings, pre_logits=True)  # Shape: (1, 384)
    student_embeddings = projection_layer(student_embeddings).squeeze().cpu().numpy()  # Shape: (512,)


print(student_embeddings)







