import os
import csv


def write_csv(result, save):
    colName = ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y']
    prefix = "submission"
    with open(os.path.join(save, prefix + ".csv"), 'w') as f:
        write = csv.writer(f)
        write.writerow(colName)
        write.writerows(result)
        f.close()
