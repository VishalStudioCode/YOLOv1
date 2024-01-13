import numpy as np
import torch
import pathlib
import cv2 as cv
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json


#bdd100k loader
class BDD100KDataset(Dataset):
    def __init__(self, image_size=448, file_path="/data/mtsysin/ipa/LaneDetection_F23/bdd100k", grid_size=7, bb_box=2, cls=13):
        #class table map
        self.classes = ["pedestrian", "rider", "car", "truck", "bus", "train",
                        "motorcycle", "bicycle", "traffic light", "traffic sign", "other vehicle","other person", "trailer"]

        self.image_size = image_size;
        self.S = grid_size;
        self.B = bb_box;
        self.C = cls;
        self.paths = []; #PATH to image
        self.boxes = []; #bounding boxes
        self.labels = []; #labels of bounding boxes
        #convert numpy array to tensor
        self.to_tensor = transforms.ToTensor();

        self.grid_x = torch.Tensor(self.S);
        self.grid_y = torch.Tensor(self.S);
        #load json file for bdd100k (might take a sec since its so big)
        bdd100k_root_path = pathlib.Path(file_path);
        train_file_path = bdd100k_root_path / "labels"/ "det_20" / "det_train.json";
        #list of dictionaries for every image
        bdd100k_json = json.load(train_file_path.open());
        for image_dict in bdd100k_json:
            #get images from training folder
            img_path = bdd100k_root_path / "images"/ "100k" / "train"/ image_dict["name"];
            if(not img_path.exists() or "labels" not in image_dict.keys()):
                continue;
            self.paths.append(img_path);
            #find number of bounding boxes, and their labels
            box = [];
            label = [];
            for img_object in image_dict["labels"]:
                box.append(list(img_object["box2d"].values()));
                label.append(self.classes.index(img_object["category"]));

            self.boxes.append(box);
            self.labels.append(label);
        self.grids = [[] for i in range(len(self.paths))];

    def __len__(self):
        #number of images
        return len(self.labels);
    def __getitem__(self, idx):
        path = self.paths[idx];
        img = cv.imread(str(path));
        img_h, img_w = img.shape[:2];
        for i in range(self.S):
            self.grid_x[i] = img_w*i//self.S;
            self.grid_y[i] = img_h*i//self.S;

        #tensor
        target = torch.zeros(self.S, self.S, self.B*5 + self.C);
        boxes = self.boxes[idx];
        counter = [[0]*self.S for i in range(self.S)];
        #get width, height for each box
        for i, box in enumerate(boxes):
            #mid point
            x, y = (box[:2] + box[2:])/2;
            S1 = len(self.grid_x[x >= self.grid_x]) -1;
            S2 = len(self.grid_y[y >= self.grid_y]) -1;
            #normalize depending on the grid
            #should investigate this to make sure it is normalized correctly
            x /= (int(self.grid_x[x >= self.grid_x][-1]) + int(self.grid_x[1]));
            y /= (int(self.grid_y[y >= self.grid_y][-1]) + int(self.grid_y[1]));
            # get width and height (2nd Point - 1st Point)
            w, h = torch.abs(box[2:] - box[:2]);
            #normalize
            w /= int(img.shape[1]);
            h /= int(img.shape[0]);
            #encode into tensor [S, S, 5*B + C]
            #encode x,y,w,h conf
            if(counter[S1][S2] < self.B):
                self.grids[idx].append((S1, S2));
                target[S1][S2][counter[S1][S2]*5:(counter[S1][S2]+1)*5] = torch.Tensor([x,y,w,h,1]);
                #confidence
                target[S1][S2][5*self.B + int(self.labels[idx][i])] = 1;
            counter[S1][S2] += 1;

        #resize image to 448
        img = cv.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB);
        
        return img, target;

