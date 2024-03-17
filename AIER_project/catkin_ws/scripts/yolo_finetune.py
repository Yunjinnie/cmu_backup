#from groundingdino.util.inference import load_model, load_image, predict, annotate
#from datasets import load_dataset
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
import cv2
import yaml


#rf = Roboflow(api_key = "Oudtx9P8vlJcoGUSPjBQ")
#project = rf.workspace("jetbot").project("petson")
#dataset = project.version(3).download("yolov8")
#!mkdir datasets
#!mv petson-3/ datasets/


def run_dino(dino, image, text_prompt='scotty', box_threshold=0.4, text_threshold=0.1):
    boxes, logits, phrases = predict(
        model = dino, 
        image = image, 
        caption = text_prompt, 
        box_threshold = box_threshold, 
        text_threshold = text_threshold
    )
    return boxes, logits, phrases
'''
model = load_model('GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 'groundingdino_swint_ogc.pth') ###

os.system('wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg') ###
image_source, image = load_image('dog.jpeg')
boxes, logits, phrases = run_dino(dino, image, text_prompt='dog')

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
sv.plot_image(annotated_frame, (8, 8))

def annotate(dino, data, data_size, data_dir):
    data = data.train_test_split(train_size=min(len(data), data_size))['train']

    image_dir = f'{data_dir}/images'
    label_dir = f'{data_dir}/labels'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    for i, d in enumerate(tqdm(data)):
        image_path = f'{image_dir}/{i:06d}.png'
        label_path = f'{label_dir}/{i:06d}.txt'
        image = d['image'].resize((640, 640))
        image.save(image_path)
        
        image_source, image = load_image(image_path)
        boxes, logits, phrases = run_dino(dino, image)

        label = ['0 ' + ' '.join(list(map(str, b))) for b in boxes.tolist()]
        label = '\n'.join(label)
        with open(label_path, 'w') as f:
            f.write(label)


data = load_dataset('COCO') # food101
annotate(dino, data['train'], 3000, 'data/train')
annotate(dino, data['validation'], 1000, 'data/valid')

config = {
    'names': ['scotty'],
    'nc': 1,
    'train': 'train/images',
    'val': 'valid/images'
}

with open('/home/orin/scotty-4/scotty-4/data.yaml', 'w') as f:
    yaml.dump(config, f)
'''

yolo = YOLO('/home/orin/yolov8_rs/yolov8m.pt') # yolo8n
yolo.train(data='/home/orin/scotty-4/scotty-4/data.yaml', epochs=10)
valid_results = yolo.val()
print(valid_results)

def run_yolo(yolo, image_url, conf=0.25, iou=0.7):
    results = yolo(image_url, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2,1,0]]
    return Image.fromarray(res)
    
#image_url = './' ### scotty
img = Image.open("")
results = model.predict(source=img, save=True)
print(results)
display(Image.open('runs/detect/predict/image0.jpg'))

success = model.export(format="onnx")
