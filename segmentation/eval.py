from pathlib import Path
import cv2
import numpy as np
from time import perf_counter
import argparse


def main(args):
    # Model path
    checkpoint_path = str(Path(args.model_path).resolve())
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise ValueError("Model path is required")

    print(f"****************Checkpoint: {checkpoint_path}****************")

    # Import model based on the model size required
    if args.model_size == "nano":
        from rfdetr import RFDETRSegNano
        model = RFDETRSegNano(pretrain_weights=checkpoint_path)
    elif args.model_size == "small":
        from rfdetr import RFDETRSegSmall
        model = RFDETRSegSmall(pretrain_weights=checkpoint_path)
    elif args.model_size == "medium":
        from rfdetr import RFDETRSegMedium
        model = RFDETRSegMedium(pretrain_weights=checkpoint_path)
    elif args.model_size == "large":
        from rfdetr import RFDETRSegLarge
        model = RFDETRSegLarge(pretrain_weights=checkpoint_path)
    else:
        raise ValueError("Invalid model size. Options: nano, small, medium, large")

    print(f"****************Model: {args.model_size}****************")

    
    # Load image
    image_dir = Path(args.image_path).resolve()

    # Load image and predict
    if image_dir is None or not Path(image_dir).exists():
        raise ValueError("Image path is required")
    
    img = cv2.imread(str(image_dir))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Getting predictions for {image_dir}")
    # Predict
    start_time = perf_counter()
    pred = model.predict(img)
    end_time = perf_counter()
    print(f"Time: {end_time - start_time} s")

    # Draw masks on the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, mask, class_id, conf in zip(pred.xyxy, pred.mask, pred.class_id, pred.confidence):
        color = np.random.randint(0, 255, 3)
        alpha = 0.3
        overlay = img.copy()
        overlay[mask] = color
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.putText(img, f"Class: {class_id}, Conf: {conf:.2f}", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default=None, help="Path to the model checkpoint", required=True)
    parser.add_argument("-i", "--image_path", type=str, default=None, help="Path to the image to predict", required=True)
    parser.add_argument("-s", "--model_size", type=str, default="small", help="Model size. Options: nano, small, medium, large. Default: small", required=False)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    main(args)
