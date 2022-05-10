import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models.bisenetv1 import BiSeNetv1
from semseg.datasets.ai4mars import ai4mars
from semseg.utils.utils import timer
import cv2
from semseg.utils.visualize import draw_text
from PIL import Image
from rich.console import Console

console = Console()


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        # get dataset classes' colors and labels
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.labels))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu')['model'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # get segmentation map (value being 0 to num_classes)

        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()

        if overlay:
            over_seg_image = (orig_img * 0.4) + (seg_image * 0.6)

        image = draw_text(seg_image, seg_map, self.labels)
        return seg_image,over_seg_image,image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = cv2.imread(img_fname)
        image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img = self.preprocess(image1)
        seg_map = self.model_forward(img)
        seg_map,over_seg_map,image = self.postprocess(torch.tensor(image1), seg_map, overlay)
        return seg_map,over_seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ai4mars.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)

    semseg = SemSeg(cfg)

    with console.status("[bright_green]Processing..."):
        if test_file.is_file():
            console.rule(f'[green]{test_file}')
            segmap,oversegmap,image = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
            oversegmap_arr=segmap.numpy()
            segmap_arr=segmap.numpy()
            segmap_arr[segmap_arr==1]=50
            segmap_arr[segmap_arr==2]=100
            segmap_arr[segmap_arr==3]=150
            cv2.imwrite('pic1.png',segmap_arr)
            cv2.imwrite('pic2.png',segmap_arr)
            im1 = image.save("geeks.jpg")

            # segmap.save(save_dir / f"{str(test_file.stem)}.png")
        else:
            files = test_file.glob('*.*')
            for file in files:
                console.rule(f'[green]{file}')
                segmap = semseg.predict(str(file), cfg['TEST']['OVERLAY'])
                segmap.save(save_dir / f"{str(file.stem)}.png")

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")
