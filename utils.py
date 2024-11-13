import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class MergeDataset:
    def __init__(self, folder1, folder2, savedir=None):
        self.folder1 = folder1
        self.folder2 = folder2
        if savedir:
            self.savedir = savedir
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
                print(f'{self.savedir} create success')

    def mergeLabels(self):
        saveLabel_dir = self.savedir + '\\labels' 
        if not os.path.exists(saveLabel_dir):
            os.makedirs(saveLabel_dir)
        labelClass = []
        for num, (label1, label2) in enumerate(zip(os.listdir(self.folder1), os.listdir(self.folder2))):
            ## copy folder1
            shutil.copy(self.folder1+f'\\{label1}', saveLabel_dir+f'\\{label1}')
            with open(self.folder1+f'\\{label1}', 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    if int(parts[0]) not in labelClass:
                        labelClass.append(int(parts[0]))

            # process folder2
            lines = []
            with open(self.folder2+f'\\{label2}', 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    if parts:
                        lbl = sorted(labelClass)[-1] + 1
                        parts[0] = f'{lbl}'
                        lines.append(' '.join(parts)+'\n')
            with open(saveLabel_dir+f'\\{label2}', 'a') as file:
                file.writelines(lines)
         

class test_yolo_result:
    '''
        Enter image and (bbox or segs) to confirm that the coordinates are correct. 
        example :
            test_yolo_result(
                image = 'folder path',
                bbox = 'folder path',
                segs = 'folder path'
            ).main()
    '''
    def __init__(self, image, bbox=None, segs=None):
        self.test_image_folder = image
        self.test_box_folder = bbox
        self.test_seg_folder = segs
        self.bbox_data = {}
        self.seg_data = {}

    def show_seg(self, coords, ax):   
        for s in range(len(coords)):
            seg = coords[s].reshape(-1, 2)
            ax.add_patch(ax.add_patch(plt.Polygon(xy=seg, color='red', alpha=0.2)))

    def show_box(self, coords, ax):
        for j in range(len(coords)):
            box = coords[j]
            x0, y0 = box[1], box[2]
            w, h = box[3] - box[1], box[4] - box[2]
            if box[0] == 0:
                ax.add_patch(ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2)))
            elif box[0] == 1:
                ax.add_patch(ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2)))  


    def process_bbox(self):
        for num, (image, box) in enumerate(zip(os.listdir(self.test_image_folder), os.listdir(self.test_box_folder))):
            imagefile = self.test_image_folder + "\\" +image
            boxfile = self.test_box_folder + '\\' + box
            h, w, _ = cv2.imread(imagefile).shape
            bbox = []
            with open(boxfile, 'r') as f:
                for line in f.readlines():
                    bbox.append(line.rstrip(' \n'))
            coords_list = []        
            for box_coords in bbox:
                
                bx_list = np.array([x for x in box_coords.split(' ')], dtype=np.float32)
                bx_list[1:] = bx_list[1:] * [w, h, w, h]
                coords_list.append(np.array([
                                            bx_list[0],
                                            bx_list[1] - bx_list[3]/2, 
                                            bx_list[2] - bx_list[4]/2, 
                                            bx_list[1] + bx_list[3]/2,
                                            bx_list[2] + bx_list[4]/2,
                                        ], dtype='uint8'))  # xywh -> xyxy
            self.bbox_data[num] = coords_list
        return True
        
    def process_segments(self):
        for num,(image, seg) in enumerate(zip(os.listdir(self.test_image_folder), os.listdir(self.test_seg_folder))):
            imagefile = self.test_image_folder + "\\" + image
            segfile = self.test_seg_folder + '\\' + seg
            h, w, _ = cv2.imread(imagefile).shape
            segment = []
            with open(segfile, 'r') as segf:
                for line in segf.readlines():
                    segment.append(line.rstrip(' \n'))
            
            segment_list = []
            for seg_coords in segment:
                coords_list = []
                sg_list = np.array([s for s in seg_coords.split(' ')], dtype=np.float32)    
                for c in sg_list[1:].reshape(-1, 2):
                    coords_list.extend([int(c[0] * w), int(c[1] * h)])
                segment_list.append(np.array(coords_list))

            self.seg_data[num] = segment_list
        return True

    def main(self, box=False, seg=False):
        seg = self.process_segments() if self.test_seg_folder else None 
        box = self.process_bbox() if self.test_box_folder else None
        for i, imageName in enumerate(os.listdir(self.test_image_folder)):
            im = self.test_image_folder + '\\' + imageName
            image = cv2.imread(im)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            if box:
                plt.title(f'{i} : {imageName}')
                self.show_box(self.bbox_data[i], plt.gca())
            if seg:
                self.show_seg(self.seg_data[i], plt.gca())
            plt.show()

class correction_dataset:
    '''
        Input image folder and label folder to correct redundant labels.
        hint : main -> show_image_with_bbox -> onclick -> check_in_box -> show_image_with_bbox -> edit_text_file
        example :
            correction_dataset(
                image='folder path',
                bbox='folder path',
                isBox = True,   <-Enter this parameter if the input is boundingbox, otherwise do not
            ).main()
    '''
    def __init__(self, image, label, isBox=False):
        self.isBox = isBox
        self.test_image_folder = image
        self.test_label_folder = label
        self.bbox_data = {}
        self.bbox_data_index = -1
        self.xy = [0,0]
        self.image = None
        self.ax = None

    def show_box(self, coords):
        '''
            The first position in the coords is the category number, 
            you can read box[0] to choose whether to separate different colors.
        '''
        for j in range(len(coords)):
            box = coords[j]
            x0, y0 = box[1], box[2]
            w, h = box[3] - box[1], box[4] - box[2]
            self.ax.add_patch(plt.Circle(xy=(x0, y0), radius=1, edgecolor='red', fill=False))
            if box[0] == 0:
                self.ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
            elif box[0] == 1:
                self.ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))

    def show_seg(self, coords):   
        for s in range(len(coords)):
            seg = coords[s].reshape(-1, 2)
            self.ax.add_patch(self.ax.add_patch(plt.Polygon(xy=seg, color='red', alpha=0.2)))

    def process_bbox(self):
        '''
            from yolo format to pyplot.Rectangle format, than save in dict
        '''
        for num, (image, box) in enumerate(zip(os.listdir(self.test_image_folder), os.listdir(self.test_label_folder))):
            imagefile = self.test_image_folder + "\\" +image
            boxfile = self.test_label_folder + '\\' + box
            h, w, _ = cv2.imread(imagefile).shape
            bbox = []
            with open(boxfile, 'r') as f:
                for line in f.readlines():
                    bbox.append(line.rstrip(' \n'))
            coords_list = []        
            for box_coords in bbox:
                
                bx_list = np.array([x for x in box_coords.split(' ')], dtype=np.float32)
                bx_list[1:] = bx_list[1:] * [w, h, w, h]
                coords_list.append(np.array([
                                            bx_list[0],
                                            bx_list[1] - bx_list[3]/2, 
                                            bx_list[2] - bx_list[4]/2, 
                                            bx_list[1] + bx_list[3]/2,
                                            bx_list[2] + bx_list[4]/2,
                                        ], dtype='uint8'))  # xywh -> xyxy
            self.bbox_data[num] = coords_list

    def process_bbox_from_segments(self):
        for num,(image, seg) in enumerate(zip(os.listdir(self.test_image_folder), os.listdir(self.test_label_folder))):
            imagefile = self.test_image_folder + "\\" + image
            segfile = self.test_label_folder + '\\' + seg
            h, w, _ = cv2.imread(imagefile).shape
            segment = []
            with open(segfile, 'r') as segf:
                for line in segf.readlines():
                    segment.append(line.rstrip(' \n'))
            
            
            coords_list = []
            for seg_coords in segment:
                segment_list = []
                sg_list = np.array([s for s in seg_coords.split(' ')], dtype=np.float32)    
                for c in sg_list[1:].reshape(-1, 2):
                    segment_list.append([[c[0] * w, c[1] * h]])
                x, y, w, h = cv2.boundingRect(np.array(segment_list, dtype=np.int32))
                coords_list.append(np.array([
                    sg_list[0],
                    x,
                    y,
                    x + w,
                    y + h,
                ]))
            self.bbox_data[num] = coords_list
        return True
    
    def Edit_text_file(self, flags):
        '''
            Find out in which file the selected bounding box is located
            flags(column number) -> Records which column of the file is currently being read
        '''
        editFile = self.test_label_folder + '\\' + os.listdir(self.test_label_folder)[self.bbox_data_index]     
        with open(editFile, 'r') as f:
            lines = f.readlines()      
        if 0 <= flags <= len(lines):
            del lines[flags]
        with open(editFile, 'w') as f:
            f.writelines(lines) 
        
        
    def check_in_box(self):
        '''
            Verify that the coordinates are in the upper left of the bounding box.
        '''
        alert = True
        for flags, coords in enumerate(self.bbox_data[self.bbox_data_index]):  ## x, y = coords[1], coords[2]
            distance = np.sqrt((self.xy[0] - coords[1])**2 + (self.xy[1] - coords[2])**2)
            if distance < 1:
                ## 選取目標準備移除
                print(f'del {self.bbox_data[self.bbox_data_index][flags]}')
                del self.bbox_data[self.bbox_data_index][flags]      
                ## 更新顯示畫面
                self.show_image_with_bbox(os.listdir(self.test_image_folder)[self.bbox_data_index])
                self.Edit_text_file(flags)
                alert = False
                break
        if alert:
            print('Please click bounding box top-left or bottom-right')

    
    def show_image_with_bbox(self, imageName):
        '''
            Reads pictures and creates sub-screen, each call clears the current sub-screen and updates it.
        '''
        im = self.test_image_folder + '\\' + imageName
        if self.image is None:
            self.image = cv2.imread(im)
            plt.figure(figsize=(8,8))
            self.ax = plt.gca()
        else:
            self.image = cv2.imread(im)
            self.ax = plt.gca()
        self.ax.clear()
        self.ax.imshow(self.image)
        self.show_box(self.bbox_data[self.bbox_data_index])
        plt.title(f'{self.bbox_data_index} : {imageName}')
        plt.draw()

    def onclick(self, event):
        if event.inaxes:
            # print(f'x : {event.xdata}, y : {event.ydata}')
            self.xy = [event.xdata, event.ydata]
            self.check_in_box()
            sleep(0.1)


    def main(self):
        self.process_bbox() if self.isBox else self.process_bbox_from_segments()
        for i, imageName in enumerate(os.listdir(self.test_image_folder)):
            self.bbox_data_index = i
            self.show_image_with_bbox(imageName)
            plt.connect('button_press_event', self.onclick) 
            plt.show()



if __name__ == "__main__":
    # merge = MergeDataset(
    #     folder1=r'D:\Admin\Desktop\labels',
    #     folder2=r'D:\Admin\Desktop\C6204_RF_DATA\C6204_RF_COMP\labels',
    #     savedir=r'D:\Admin\Desktop\merge_result',
    # ).mergeLabels()

    # test_yolo_result(
    #     image=r'D:\Admin\Desktop\C6204_RF_DATA\images',
    #     bbox=r'D:\Admin\Desktop\C6204_RF_DATA\C6204_RF_COMP\labels',
    # ).main()

    correction_dataset(
        image=r'D:\Admin\Desktop\labels_E_test\U601',
        label=r'D:\Admin\Desktop\labels_E_test\box_labels_E',
        isBox=True
    ).main()