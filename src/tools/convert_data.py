import os
import json
import cv2
import base64
import csv
import argparse


CLASSES = {
    '0': 'CMS1216',
    '1': 'CMS1719',
    '2': 'CSH1621',
    '3': 'CTS21',
    '4': 'CTS1719',
    '5': 'GGS1620',
    '6': 'GGS21',
    '7': 'GGS20',
    '8': 'HAS1115',
    '9': 'HAS20',
    '10': 'HGS1116',
    '11': 'HGV1820',
    '12': 'HIH1619',
    '13': 'HSS0409',
    '14': 'HSS1014',
    '15': 'HSS1920',
    '16': 'KCV1520',
    '17': 'KCV21',
    '18': 'KKS1015',
    '19': 'KKS20',
    '20': 'KKS1620',
    '21': 'KMS20',
    '22': 'KMH0410',
    '23': 'KMH1116',
    '24': 'KRH1217',
    '25': 'KSS1519',
    '26': 'KSS20',
    '27': 'KSS1418',
    '28': 'KSS1620',
    '29': 'KSS1719',
    '30': 'RSS1518',
    '31': 'RXS20',
    '32': 'SKS1920',
    '33': 'STS1620'
    }

REVERSED_CLASSES = {v:k for k,v in CLASSES.items()}


def yolo2labelme(name, annots, imgDir, saveDir):
    image = cv2.imread(os.path.join(imgDir, name + '.png'))
    imageHeight, imageWidth, _ = image.shape
    labelme = {}
    labelme.update({"version": "5.2.1"})
    labelme.update({"flags":{}})
    preAnnot = []
    for annot in [i.strip('\n') for i in annots]:
        annot = annot.split(' ')
        label = CLASSES[annot[0]]
        annot = list(map(float, annot))
        topLeft = [imageWidth*(annot[1] - annot[3] / 2), imageHeight*(annot[2] - annot[4] / 2)]
        bottomRight = [imageWidth*(annot[1] + annot[3] / 2), imageHeight*(annot[2] + annot[4] / 2)]
        preAnnot.append({"label": label, "points": [topLeft, bottomRight], "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}})
    labelme.update({"shapes": preAnnot})
    labelme.update({"imagePath": name + ".png"})
    with open(os.path.join(imgDir, name + '.png'), 'rb') as f:
        s = base64.b64encode(f.read())
        labelme.update({"imageData": s.decode('utf-8')})
    labelme.update({"imageHeight": imageHeight})
    labelme.update({"imageWidth": imageWidth})
    with open(os.path.join(saveDir, name + ".json"), "w+") as file:
        json.dump(labelme, file)


def yolo2submission(imgDir, name, annots):
    image = cv2.imread(os.path.join(imgDir, name + '.png'))
    imageHeight, imageWidth, _ = image.shape
    lst = []
    for annot in [i.strip('\n') for i in annots]:
        annot = annot.split(' ')
        label = annot[0]
        conf = annot[-1]
        annot = list(map(float, annot))
        x0, y0 = [imageWidth*(annot[1] - annot[3] / 2), imageHeight*(annot[2] - annot[4] / 2)]
        x2, y2 = [imageWidth*(annot[1] + annot[3] / 2), imageHeight*(annot[2] + annot[4] / 2)]
        x1, y1 = x2, y0
        x3, y3 = x0, y2
        x0, y0, x1, y1, x2, y2, x3, y3 = list(map(str, (x0, y0, x1, y1, x2, y2, x3, y3)))
        lst.append((name + '.png', label, conf, x0, y0, x1, y1, x2, y2, x3, y3))
    return lst
        

def labelme2yolo(savePath, json):

    bboxes = json['shapes']
    imageWidth = json['imageWidth']
    imageHeight = json['imageHeight']
    with open(savePath, 'w+') as txt:
        for bbox in bboxes:
            label = REVERSED_CLASSES[bbox['label']]
            topLeft, bottomRight = bbox['points']
            if topLeft[0] > bottomRight[0]:
                topLeft, bottomRight = bottomRight, topLeft
            w = abs(bottomRight[0] - topLeft[0]) / imageWidth
            h = abs(bottomRight[1] - topLeft[1]) / imageHeight
            x = topLeft[0] / imageWidth + w / 2
            y = topLeft[1] / imageHeight + h / 2
            txt.write('{} {} {} {} {}'.format(label, x, y, w, h))
            txt.write('\n')
        txt.close()


def processYolo2labelme(yoloTxt, imageDir, saveJson):
    for file in [os.path.join(yoloTxt, i) for i in os.listdir(yoloTxt) if i.endswith(".txt")]:
        name = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            print(file)
            predictions = f.readlines()
            yolo2labelme(name=name, annots=predictions, imgDir=imageDir, saveDir=saveJson)
        f.close()
    print("Done")


def processLabelme2yolo(labelmeJson, saveTxt):

    for file in [os.path.join(labelmeJson, i) for i in os.listdir(labelmeJson) if i.endswith(".json")]:
        name = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            print(file)
            annotations = json.load(f)
            labelme2yolo(os.path.join(saveTxt, name + '.txt'), annotations)


def processYolo2submission(yoloTxt, imageDir, saveTxt):
    result = []
    colName = ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y']
    for file in [os.path.join(yoloTxt, i) for i in os.listdir(yoloTxt) if i.endswith(".txt")]:
        name = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            print(file)
            predictions = f.readlines()
            result.extend(yolo2submission(imgDir=imageDir, name=name, annots=predictions))
        f.close()
    with open(os.path.join(saveTxt, "yolov5_augmented.csv"), 'w+') as ftxt:
        write = csv.writer(ftxt)
        write.writerow(colName)
        write.writerows(result)
    ftxt.close()
    
    print("Done")



def main():

    parse = argparse.ArgumentParser(description='Car type detection')
    parse.add_argument('--images', type=str, default='', help='Path to the image folder')
    parse.add_argument('--labels', type=str, default='', help='Path to labels folder')
    parse.add_argument('--save', type=str, default='', help='Path to save converted labels')
    parse.add_argument('--task', choices=['yolo2labelme', 'labelme2yolo', 'yolo2submission'], default='', help='Convert task')

    args = parse.parse_args()
    ### task yolo2labelme: converts labels from yolo .txt to labelme .json files
    if args.task == 'yolo2labelme':
        processYolo2labelme(yoloTxt=args.labels, imageDir=args.images, saveJson=args.save)

    ### task labelme2tolo: converts labels from labelme .json to yolo .txt files
    elif args.task == 'labelme2yolo':
        processLabelme2yolo(labelmeJson=args.labels)

    ### task yolo2submission: converts results from yolo .txt to submission.csv file
    if args.task == 'yolo2submission':
        processYolo2submission(yoloTxt=args.labels, imageDir=args.images, saveTxt=args.save)


if __name__ == '__main__':
    main()