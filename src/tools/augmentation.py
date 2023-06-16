import random
import albumentations as A
import cv2
import os
import argparse

def tranformer(imagePath, labelsPath, transform, savePath):
    print(imagePath)
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(labelsPath, 'r') as txt:
        data = txt.readlines()
        txt.close()
    data = [i.strip('\n').split(' ') for i in data]
    bboxes = [list(map(float, i[1:])) for i in data]
    classLabels = [i[0] for i in data]
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=classLabels)

        transformedImage = transformed['image']
        transformedBboxes = transformed['bboxes']
        transformedClassLabels = transformed['class_labels']

        prefix = 'augmentation'
        fileName = prefix + '_' + str(random.randint(10000, 15000)).zfill(5)

        # savePath: path to save results with 2 subfolder 'images' and 'labels' 

        with open(os.path.join(savePath, 'labels', fileName + '.txt'), 'w+') as ftxt:
            for bbox, label in zip(transformedBboxes, transformedClassLabels):
                ftxt.write(label + ' ' + ' '.join(list(map(str, bbox))))
                ftxt.write('\n')

        cv2.imwrite(os.path.join(savePath, 'images', fileName + '.png'), transformedImage)
        print("-"*20 + fileName + "-"*20)
    except ValueError as e:
        pass


def main():
    parse = argparse.ArgumentParser(description='Car type detection')
    parse.add_argument('--images', type=str, default='', help='Path to images')
    parse.add_argument('--labels', type=str, default='', help='Path to labels')
    parse.add_argument('--save', type=str, default='test_augmentation', help='Path to save result')

    args = parse.parse_args()

    transform = A.Compose([
        A.HorizontalFlip(p=0.8),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Blur(p=0.2),
        # A.BBoxSafeRandomCrop()

    ], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels'], min_area=20000, min_visibility=0.7))

    if os.path.isfile(args.images) and os.path.isfile(args.labels):
        tranformer(imagePath=args.images, labelsPath=args.labels, transform=transform, savePath=args.save)
    elif os.path.isdir(args.images) and os.path.isdir(args.labels):
        assert len(os.listdir(args.images)) == len(os.listdir(args.labels)), "Number of images and labels did not match {} vs {}".format(len(os.listdir(args.images)), len(os.listdir(args.labels)))
        for fileName in [i.split('.')[0] for i in os.listdir(args.images)]:
            tranformer(imagePath=os.path.join(args.images, fileName + '.png'), labelsPath=os.path.join(args.labels, fileName + '.txt'), transform=transform, savePath=args.save)


if __name__ == '__main__':
    main()