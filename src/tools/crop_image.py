import cv2
import os

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

def cropImageAndSave(image, labels, savePath, fileName):
    imageHeight, imageWidth, _ = image.shape
    c = 0
    for label in labels:
        gt = CLASSES[label[0]]
        if os.path.exists(os.path.join(savePath, gt)):
            pass
        else:
            os.mkdir(os.path.join(savePath, gt))
        label = list(map(float, label))
        topLeft = list(map(int, (imageWidth*(label[1] - label[3] / 2), imageHeight*(label[2]- label[4] / 2))))
        bottomRight = list(map(int, (imageWidth*(label[1] + label[3] / 2), imageHeight*(label[2] + label[4] / 2))))
        crop = image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        cv2.imwrite(os.path.join(savePath, gt, fileName + '_0{}.png'.format(str(c))), crop)
        c += 1


if __name__ == '__main__':

    savePath = '/data/son/dataset/crop_test'
    data = '/data/son/dataset/origin_testset'
    listFile = [os.path.join(data + '/labels', i) for i in os.listdir(data + '/labels') if i.endswith('.txt')]
    gtWithFilename = {}
    for file in listFile:
        fileName = file.split('/')[-1].split('.')[0]
        with open(file) as f:
            labels = f.readlines()
            labels = [i.strip('\n').split(' ') for i in labels]
            gtWithFilename.update({fileName: labels})
            f.close()

    for i in gtWithFilename.items():
        # if 'syn' in i[0]:
            # image = cv2.imread(os.path.join(data + '/images', i[0] + '.png'))
            # cropImageAndSave(image=image, labels=i[1], savePath=savePath, fileName=i[0])
        image = cv2.imread(os.path.join(data + '/images', i[0] + '.png'))
        cropImageAndSave(image=image, labels=i[1], savePath=savePath, fileName=i[0])