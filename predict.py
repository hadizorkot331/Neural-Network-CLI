import json
import torch
import argparse
from PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="CLI for using a model to get predictions")

    parser.add_argument("image_path", help="Path to input image", required=True)
    parser.add_argument("checkpoint_path", help="Path to model checkpoint", required=True)
    parser.add_argument("--category_names" help="Path to json mapping of category indices to names")
    parser.add_argument("--top_k" help="Top k predictions")
    parser.add_argument("--gpu", action="store_true", help="Choose to predict on GPU")

    args = parser.parse_args()
    return args

def load_model(checkpoint):
    model_info = torch.load_model(checkpoint)
    model = model_info["model"]
    model.classifier = model_info["classifier"]
    model.load_state_dict(model_info["state_dict"])

    return model

def process_image(image):
    image = Image.open(image)
    image.thumbnail((256,256), Image.ANTIALIAS)
    
    width, height = image.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    np_image = np_image / [255,255,255]
    np_image = (np_image - np.array([0.485, 0.456, 0.406]))/ np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2,0,1))

    return np_image


def make_prediction(image, model, device, topk=3):
    processed_image = torch.from_numpy(image)
    processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    with torch.no_grad():
        processed_image = processed_image.to(device)
        model = model.to(device)
        outputs = model(processed_image)
        top_ps, top_class = torch.exp(outputs).topk(topk)
    
        return zip(top_ps[0].tolist(), top_class[0].tolist())
    
def get_categories(category_names):
    if category_names:
        with open(category_names) as f:
            names = json.loads(f.read())
            return names
    return None

def print_prediction(prediction, category_names):
    for prob, cls in prediction:
        if category_names:
            cls = category_names[cls]
        
        print(f"Class: {cls} with a confidence of {prob*100:.3f}")

def main():
    args = get_args()
    image = process_image(args.image_path)
    model = load_model(args.checkpoint_path)
    device = "cuda" if args.gpu else "cpu"
    top_k = int(top_k) if args.top_k is not None else 3
    prediction = make_prediction(image, model, device, top_k)
    categories = get_categories(args.category_names)
    print_prediction(prediction, categories)

main()