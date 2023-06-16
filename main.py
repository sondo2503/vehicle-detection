from src.detection.detector import Detector
from src.tools.write_csv import write_csv

import csv
import cv2
import glob
from decouple import config
import argparse 

DETECTOR = Detector(config('model_weights'))

def main():
    parse = argparse.ArgumentParser(description='Car type detection')
    parse.add_argument('--image', type=str, default='', help='Path to the image')
    parse.add_argument('--folder', type=str, default='', help='Path to image folder')
    parse.add_argument('--save_csv', type=bool, default=False, help='Save csv submission')

    args = parse.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        preds = DETECTOR.detect_vehicle(image, thresh=0.8)
        print(preds)
    elif args.folder:
        results = []
        for file in glob.glob(args.folder + '/*.png'):
            file_name = file.split("/")[-1]
            image = cv2.imread(file)
            preds = DETECTOR.detect_vehicle(image, file_name=file_name)
            print(preds)
            results.extend(preds)

        if args.save_csv:
            write_csv(result=results, save='./')

            

        
if __name__ == '__main__':
    main()
