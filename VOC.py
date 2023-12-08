import numpy as np
import torch
import pathlib
import cv2 as cv
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VocDataset(Dataset):
    def __init__(self, image_size=448, file_path="data/voc2007.txt", grid_size=7, bb_box=2, cls=20) -> None:
        self.image_size = image_size;
        self.S = grid_size;
        self.B = bb_box;
        self.C = cls;
        # convert images to 448
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean_rgb, dtype=np.float32)

        self.to_tensor = transforms.ToTensor();

        # does not work because image is not 448 yet
        # self.grids = torch.Tensor([i*(image_size/self.S) for i in range(self.S)]);
        self.grid_x = torch.Tensor(self.S);
        self.grid_y = torch.Tensor(self.S);
        with open(file_path, "r") as file:
            lines = file.readlines();

        self.boxes = [];
        self.labels = [];
        self.paths = [];
        self.grids = [[]for i in range(len(lines))];
        for line in lines:
            split = line.strip().split();
            file_name = split[0];
            path = pathlib.Path("data/voc2007/VOCdevkit/VOC2007/JPEGImages") / file_name;
            # some of these images in the txt don't exist
            if(not path.exists()):
                continue;
            self.paths.append(path);
            # number of bounding boxes depending on the img in dataset
            num_boxes = len(split[1:])//5;
            box = [];
            label = [];
            for i in range(num_boxes):
                x1, y1, x2, y2 = [float(element) for element in split[i *5 +1:(i*5 + 5)]];
                # class
                c  = int(split[5*i + 5]);
                box.append([x1,y1,x2,y2]);
                label.append(c);
            
            self.boxes.append(torch.Tensor(box));
            self.labels.append(torch.LongTensor(label));
    # length of dataset
    def __len__(self):
        return len(self.labels);
    # get item
    def __getitem__(self, idx):
        path = self.paths[idx];
        img = cv.imread(str(path));
        img_h, img_w = img.shape[:2]
        for i in range(self.S):
            self.grid_x[i] = img_w*i//self.S;
            self.grid_y[i] = img_h*i//self.S;
        # tensor
        target = torch.zeros(self.S, self.S, self.B*5 + self.C);
        # [x1,y1, x2, y2] --> for n boxes
        boxes = self.boxes[idx];
        # encode counter
        counter =[[0]*self.S for i in range(self.S)];
        # get width,height for each box
        for i, box in enumerate(boxes):
            # get mid point
            x, y = (box[:2] + box[2:])/2
            S1 = len(self.grid_x[x >= self.grid_x]) -1;
            S2 = len(self.grid_y[y >= self.grid_y]) -1;
            # normalize depending on the grid
            x /= (int(self.grid_x[x >= self.grid_x][-1]) + int(self.grid_x[1]));
            y /= (int(self.grid_y[y >= self.grid_y][-1]) + int(self.grid_y[1]));
            # print(x,"\t",y);
            # get width and height (2nd Point - 1st Point)
            w, h = (box[2:] - box[:2])
            # normalize
            w /= int(img.shape[1]);
            h /= int(img.shape[0]);
            # encode into tensor [S, S, 5*B + C]
            # encode x,y,w,h,conf
            if(counter[S1][S2] < self.B):
                self.grids[idx].append((S1,S2));
                target[S1][S2][counter[S1][S2]*5:(counter[S1][S2]+1)*5] = torch.Tensor([x,y,w,h,1]);
                # classifier box
                target[S1][S2][5*self.B + int(self.labels[idx][i])] = 1;
            counter[S1][S2] +=1;
            # print(S1,"\t", S2)
        # resize the image to 448
        img = cv.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB);
        # normalize
        # img = (img - self.mean) / 255.0;
        img = self.to_tensor(img);

        return img, target;


    def segmentation(self, index):
        path = self.paths[index];
        img = cv.imread(str(path.absolute()));
        boxes = self.boxes[index];
        # labels = self.classes[index];
        for box in boxes:
            # tensor --> numpy
            # draw the bounding box on image
            pt1 = tuple([int(element) for element in box.numpy()[:2]]);
            pt2 = tuple([int(element) for element in box.numpy()[2:]])
            cv.rectangle(img, pt1=pt1, pt2=pt2,color=(0,0,255),thickness=1);
        
        cv.imwrite("testimage.jpeg", img);
            


if(__name__ == "__main__"):
    dataset = VocDataset();
    dataset.segmentation(20);