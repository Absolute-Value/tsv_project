import argparse
from io import BytesIO
import base64
from dataset.hico import build
from tqdm import tqdm

parser = argparse.ArgumentParser(description="hico")
parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
parser.add_argument("--data_path", default="/data/dataset/HICO-DET/", type=str)
parser.add_argument("--mode", default="train", type=str)
args = parser.parse_args()

def to_tsv(mode='train'):
    dataset = build(mode, args)
    actions = dataset.get_actions()
    COCO_CLASSES = dataset.COCO_CLASSES
    print(mode, len(dataset))

    targets = ""
    for i, (img, anno) in enumerate(tqdm(dataset, desc="Roading dataset")):
        target = f"{i+1}\t"
        
        hois = anno["hois"]
        boxes = anno["boxes"]
        labels = anno["labels"]
        img_buffer = BytesIO()
        img.save(img_buffer, format='png')#img.format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data) # bytes
        base64_str = base64_str.decode("utf-8") # str
        target += base64_str+'\t'

        for (hum_id, obj_id, hoi_id) in hois:
            hum_bb = boxes[hum_id]
            hum_label = labels[hum_id]
            hum_name = COCO_CLASSES[hum_label]
            
            obj_bb = boxes[obj_id]
            obj_label = labels[obj_id]
            obj_name = COCO_CLASSES[obj_label]
            
            hoi_name = actions[hoi_id]
            
            target += f"{hum_bb[0]},{hum_bb[1]},{hum_bb[2]},{hum_bb[3]},{hoi_id},{hoi_name},"
            target += f"{obj_bb[0]},{obj_bb[1]},{obj_bb[2]},{obj_bb[3]},{obj_label},{obj_name}"
            
            target += "&&"
                
        target = target[:-2]
        target += f'\n'
        
        targets += target

    out_path = f"outputs/hico-det_{mode}.tsv"
    
    print(f'writing to {out_path}...')
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(targets)
    print('done')
        
to_tsv(args.mode)
