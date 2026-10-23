from pathlib import Path
import argparse

import cv2
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "shoplifting_3dcnn.onnx"

NUM_FRAMES = 32
FRAME_WIDTH = 128
FRAME_HEIGHT = 104
THRESHOLD = 0.525

CLASS_NAMES = {
    0: "Normal shopping",
    1: "Shoplifting"
}


def sample_video(video_path):
    """Sample 32 uniformly spaced RGB frames from an MP4 video."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("The video contains no readable frames.")

    indices = np.linspace(
        0, total_frames - 1, NUM_FRAMES
    ).round().astype(np.int64)

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success:
            cap.release()
            raise ValueError(f"Could not read frame {index}.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA
        )
        frames.append(frame)

    cap.release()

    # Convert (T, H, W, C) into (1, T, C, H, W).
    video = np.stack(frames).astype(np.float32) / 255.0
    video = np.transpose(video, (0, 3, 1, 2))
    video = np.expand_dims(video, axis=0)

    return np.ascontiguousarray(video, dtype=np.float32)


def predict_video(video_path):
    """Return the predicted class and shoplifting probability."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    video = sample_video(video_path)

    logits = session.run(
        ["logits"],
        {"video": video}
    )[0]

    # Convert logits to probabilities with a stable softmax.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    shoplifting_probability = float(probabilities[0, 1])
    predicted_class = int(
        shoplifting_probability >= THRESHOLD
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "shoplifting_probability": shoplifting_probability,
        "threshold": THRESHOLD,
        "sampled_frames": NUM_FRAMES,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a shopping video using ONNX Runtime."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to an MP4 video."
    )
    args = parser.parse_args()

    result = predict_video(args.video)

    print("Predicted class:", result["predicted_class"])
    print(
        "Shoplifting probability:",
        f"{result['shoplifting_probability']:.4f}"
    )
    print("Decision threshold:", result["threshold"])
    print("Sampled frames:", result["sampled_frames"])
