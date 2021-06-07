import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
from collections import OrderedDict
import csv


def rotate_image(list_of_pixels, degree):
    orig_pixel_map = [tuple(list_of_pixels[x:x + 3]) for x in range(0, len(list_of_pixels), 3)]
    width = 28
    height = 28
    mode = "RGB"
    new_image = Image.new(mode, (width, height))
    new_pixel_map = new_image.load()
    for x in range(width):
        for y in range(height):
            new_pixel_map[x, y] = orig_pixel_map[x + y * 28]

    new_image.rotate(degree)

    return to_flat_list(new_image)


def enhance_image(list_of_pixels, brightness):
    orig_pixel_map = [tuple(list_of_pixels[x:x + 3]) for x in range(0, len(list_of_pixels), 3)]
    width = 28
    height = 28
    mode = "RGB"
    new_image = Image.new(mode, (width, height))
    new_pixel_map = new_image.load()
    for x in range(width):
        for y in range(height):
            new_pixel_map[x, y] = orig_pixel_map[x + y * 28]

    enhancer = ImageEnhance.Brightness(new_image)

    factor = 1.5
    im_output = enhancer.enhance(factor)

    return to_flat_list(im_output)


def to_flat_list(image):
    flat_list = np.array(image)
    flat_list = [item for sublist in flat_list for item in sublist]
    flat_list = [item for sublist in flat_list for item in sublist]
    return flat_list


def group_list(lst):
    res = [(el, lst.count(el)) for el in lst]
    return list(OrderedDict(res).items())


def create_csv(saveAddr, rows, header):
    file = open(saveAddr, 'w', newline="")

    with file:
        write = csv.writer(file)
        write.writerow(header)
        for row in rows:
            write.writerow(row)

    file.close()


def get_skin_list_element_index(list, taget_index_count, target_label):
    index_count = 0
    for index in range(0, len(list)):
        label = list[index][-1]
        if label == target_label:
            if taget_index_count == index_count:
                return index
            index_count += 1
    raise Exception("Sorry, no target index count found")


def to_statistic_array(dx, dx_type, sex, localization):
    statistic = []
    statistic.append(dx)
    statistic.append(dx_type)
    statistic.append(sex)
    statistic.append(localization)
    return statistic


df = pd.read_csv(r'hmnist_28_28_RGB.csv')

labels = df["label"].to_list()
print(group_list(labels))

values = df.values

df_metadata = pd.read_csv(r'HAM10000_metadata.csv')
statistic_header = ["dx", "dx_type", "sex", "localization"]
dx = df_metadata["dx"].to_list()
dx_type = df_metadata["dx_type"].to_list()
sex = df_metadata["sex"].to_list()
localization = df_metadata["localization"].to_list()

statistic_values = []
new_values = []
count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_6 = 0
for index in range(0, len(values)):
    label = values[index][-1]
    if label == 0 and count_0 < 800:
        new_values.append(values[index])
        statistic_values.append(to_statistic_array(dx[index], dx_type[index], sex[index], localization[index]))
        count_0 += 1
    elif label == 1 and count_1 < 800:
        new_values.append(values[index])
        statistic_values.append(to_statistic_array(dx[index], dx_type[index], sex[index], localization[index]))
        count_1 += 1
    elif label == 2 and count_2 < 800:
        new_values.append(values[index])
        statistic_values.append(to_statistic_array(dx[index], dx_type[index], sex[index], localization[index]))
        count_2 += 1
    elif label == 3 and count_3 < 800:
        new_values.append(values[index])
        statistic_values.append(to_statistic_array(dx[index], dx_type[index], sex[index], localization[index]))
        count_3 += 1

new_values_labels = [row[-1] for row in new_values]
group_new_values_labels = group_list(new_values_labels)
print(group_new_values_labels)

count_3 = group_new_values_labels[2][1]
count_3_temp = 0
while count_3_temp < 115:
    item_index = get_skin_list_element_index(new_values.copy(), count_3_temp, 3)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_3 += 1
    count_3_temp += 1

count_5 = group_new_values_labels[4][1]
count_5_temp = 0
while count_5_temp < 142:
    item_index = get_skin_list_element_index(new_values.copy(), count_5_temp, 5)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_5 += 1
    count_5_temp += 1

count_1 = group_new_values_labels[5][1]
count_1_temp = 0
while count_1_temp < 100:
    item_index = get_skin_list_element_index(new_values.copy(), count_1_temp, 1)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)

statistic_values.append(
    to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
count_1 += 1
count_1_temp += 1

count_0 = group_new_values_labels[6][1]
count_0_temp = 0
while count_0_temp < 327:
    item_index = get_skin_list_element_index(new_values.copy(), count_0_temp, 0)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_0 += 1
    count_0_temp += 1

new_values_labels = [row[-1] for row in new_values]
group_new_values_labels = group_list(new_values_labels)

count_3 = group_new_values_labels[2][1]
count_3_temp = 0
while count_3_temp < 115:
    item_index = get_skin_list_element_index(new_values.copy(), count_3_temp, 3)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 270)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_3 += 1
    count_3_temp += 1

count_5 = group_new_values_labels[4][1]
count_5_temp = 0
while count_5_temp < 142:
    item_index = get_skin_list_element_index(new_values.copy(), count_5_temp, 5)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_5 += 1
    count_5_temp += 1

count_1 = group_new_values_labels[5][1]
count_1_temp = 0
while count_1_temp < 100:
    item_index = get_skin_list_element_index(new_values.copy(), count_1_temp, 1)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_1 += 1
    count_1_temp += 1

count_0 = group_new_values_labels[6][1]
count_0_temp = 0
while count_0_temp < 146:
    item_index = get_skin_list_element_index(new_values.copy(), count_0_temp, 0)
    label = new_values[item_index][-1]
    new_image = rotate_image(new_values[item_index][0:-1], 90)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_0 += 1
    count_0_temp += 1

new_values_labels = [row[-1] for row in new_values]
group_new_values_labels = group_list(new_values_labels)
print(group_new_values_labels)

count_3 = group_new_values_labels[2][1]
count_3_temp = 0
while count_3_temp < 115:
    item_index = get_skin_list_element_index(new_values.copy(), count_3_temp, 3)
    label = new_values[item_index][-1]
    new_image = enhance_image(new_values[item_index][0:-1], 0.5)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_3 += 1
    count_3_temp += 1

count_5 = group_new_values_labels[4][1]
count_5_temp = 0
while count_5_temp < 142:
    item_index = get_skin_list_element_index(new_values.copy(), count_5_temp, 5)
    label = new_values[item_index][-1]
    new_image = enhance_image(new_values[item_index][0:-1], 0.5)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_5 += 1
    count_5_temp += 1

new_values_labels = [row[-1] for row in new_values]
group_new_values_labels = group_list(new_values_labels)

count_3 = group_new_values_labels[2][1]
count_3_temp = 0
while count_3_temp < 115:
    item_index = get_skin_list_element_index(new_values.copy(), count_3_temp, 3)
    label = new_values[item_index][-1]
    new_image = enhance_image(new_values[item_index][0:-1], 0.8)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_3 += 1
    count_3_temp += 1

count_5 = group_new_values_labels[4][1]
count_5_temp = 0
while count_5_temp < 90:
    item_index = get_skin_list_element_index(new_values.copy(), count_5_temp, 5)
    label = new_values[item_index][-1]
    new_image = enhance_image(new_values[item_index][0:-1], 0.8)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_5 += 1
    count_5_temp += 1

new_values_labels = [row[-1] for row in new_values]
group_new_values_labels = group_list(new_values_labels)

count_3 = group_new_values_labels[2][1]
count_3_temp = 0
while count_3_temp < 110:
    item_index = get_skin_list_element_index(new_values.copy(), count_3_temp, 3)
    label = new_values[item_index][-1]
    new_image = enhance_image(new_values[item_index][0:-1], 1.2)
    rotate_image_list_of_pixels = new_image
    rotate_image_list_of_pixels.append(label)
    new_values.append(rotate_image_list_of_pixels)
    statistic_values.append(
        to_statistic_array(dx[item_index], dx_type[item_index], sex[item_index], localization[item_index]))
    count_3 += 1
    count_3_temp += 1

header = df.head()
create_csv("dataset_800.csv", new_values, header)
create_csv("statistic_800.csv", statistic_values, statistic_header)

