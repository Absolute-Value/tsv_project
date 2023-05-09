import os
import argparse
from pycocotools.coco import COCO
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="coco")
parser.add_argument("--data_path", default="/data/dataset/v-coco/coco/", type=str)
parser.add_argument("--mode", default="train", type=str)
args = parser.parse_args()

class GetCOCO():
    def __init__(self, mode='train'):
        self.mode = mode
        json_path = os.path.join(args.data_path, 'annotations', f'instances_{mode}2014.json')
        self.imgs_path = os.path.join(args.data_path, 'images', f'{mode}2014')
        
        self.coco = COCO(annotation_file=json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
                
    def to_tsv(self):
        coco = self.coco
        print(self.mode, len(self.ids))
        outputs = ""
        loop = tqdm(self.ids, desc="Roading dataset")
        for i, img_id in enumerate(loop):
            
            imgs = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(img_id)
            targets = coco.loadAnns(ann_ids)
            targets = coco.loadAnns(ann_ids)
            
            image_name = imgs['file_name']
            image_path = os.path.join(self.imgs_path, image_name)
            image = Image.open(image_path).convert("RGB")
            
            img_buffer = BytesIO()
            image.save(img_buffer, format='png')#img.format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data) # bytes
            base64_str = base64_str.decode("utf-8") # str
            output = f"{i+1}\t{base64_str}\t"
            
            for target in targets:
                cat_id = target['category_id']
                cat_name = coco.cats[cat_id]['name']
                x, y, w, h = target['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
                output += f'{x1},{y1},{x2},{y2},{cat_id},{cat_name}&&'
                
            output = output[:-2]
            output += '\n'
            
            outputs += output
        
        out_path = f"outputs/hico-det_{self.mode}.tsv"
        print(f'writing to {out_path}...')
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(outputs)
        print('done')

coco = GetCOCO(args.mode)
coco.to_tsv()