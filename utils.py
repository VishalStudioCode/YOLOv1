from model import YoloV1
import pathlib
import torch
from VOC import VocDataset
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

# VOC class names and BGR color.
VOC_CLASS_BGR = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}
# load model
PATH = pathlib.Path(".") / "architecture" / "state.pt";
model = YoloV1();
model.load_state_dict(torch.load(str(PATH.absolute())));
transform = transforms.ToTensor();

# *****
#  we assume 448 by 448 img
# split in 7 grids
# and 2 bboxes
def visualize_detect(idx):


    if(isinstance(idx, str)):
        path = pathlib.Path(idx);
        if(not path.exists()):
            return
        images = [];
        for img_path in path.iterdir():
            img = cv.imread(str(img_path.absolute()));
            img_448 = cv.resize(img, (448,448),interpolation=cv.INTER_LINEAR);
            img_tensor = transform(img_448)
            detection = model(img_tensor.view(-1,*img_tensor.shape)).view(7,7,30);
            for S1 in range(0,7):
                for S2 in range(0,7):
                    if(detection[S1][S2][4] <= 0.225 ):
                        continue;
                    x1, y1 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 - detection[S1][S2][2:4]*224;
                    x2, y2 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 + detection[S1][S2][2:4]*224;
                    index_class = (detection[S1][S2][10:] == torch.max(detection[S1][S2][10:])).nonzero(as_tuple=True)[0];
                    # print(x1,y1,x2,y2)
                    key = list(VOC_CLASS_BGR.keys())[int(index_class)]
                    cv.rectangle(img_448,(int(x1), int(y1)),(int(x2),int(y2)),color=VOC_CLASS_BGR[key],thickness=1);
                    cv.putText(img_448, key ,(int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX,0.75,color=VOC_CLASS_BGR[key],thickness=2)
                    # check the target to see if there is another box in the grid
                    # if(target[S1][S2][9] != 0):
                    #     x2, y2 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 + detection[S1][S2][2:4]*448;
                    # 7x7x30
                cv.imwrite(f"prediction{img_path.name[:-4]}.jpg", img_448);
    else:
        # get image
        dataset = VocDataset();
        img_448, target = dataset.__getitem__(idx);
    # send image through the model
        detection = model(img_448.view(-1,*img_448.shape)).view(7,7,30);

        img = cv.imread(str(dataset.paths[idx]));
        img_448 = cv.resize(img,(448,448), interpolation=cv.INTER_LINEAR);
        for S1, S2 in dataset.grids[idx]:
            # print(S1, S2);
            if(S1 != -1 and S2 != -1):
                # bounding box 1
                x1, y1 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 - detection[S1][S2][2:4]*224;
                x2, y2 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 + detection[S1][S2][2:4]*224;
                index_class = (detection[S1][S2][10:] == torch.max(detection[S1][S2][10:])).nonzero(as_tuple=True)[0];
                # print(x1,y1,x2,y2)
                key = list(VOC_CLASS_BGR.keys())[int(index_class)]
                cv.rectangle(img_448,(int(x1), int(y1)),(int(x2),int(y2)),color=VOC_CLASS_BGR[key],thickness=1);
                cv.putText(img_448, key ,(int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX,0.75,color=VOC_CLASS_BGR[key],thickness=2)
                # check the target to see if there is another box in the grid
                # if(target[S1][S2][9] != 0):
                #     x2, y2 = (detection[S1][S2][:2] + torch.tensor([S1, S2]))*64 + detection[S1][S2][2:4]*448;
        # 7x7x30
        cv.imwrite(f"prediction{idx}.jpg", img_448);
    # return detection;




