import config as conf
import os
import errno

import scipy.io
import scipy.misc

import skimage
import skimage.io
import skimage.transform

import numpy as np

def time_to_str(time):
    t_in_min = time / 60.0
    t_in_hours = time / (60.0*60.0)

    frac_min = int((t_in_hours - float(int(t_in_hours)))*60.0)
    frac_sec = int((t_in_min - float(int(t_in_min))) * 60.0)
    return str(int(t_in_hours)) + 'h ' + str(frac_min) + 'm ' + str(frac_sec) + 's'


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, between_01=False, substract_mean=False, output_size=224):
    # load image
    img = skimage.io.imread(conf.project_path + conf.images_path + path)
    #img = img / 255.0
    #assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (output_size, output_size))

    if between_01==False :
        resized_img = resized_img * 255.0

    avg = 0
    if substract_mean==True:
        avg = np.sum(resized_img) / (output_size*output_size*3.0)
        resized_img -= avg

    return resized_img, avg


def save_image(path, name, images, to255=False, avg=0):
    # Output should add back the mean.
    #image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    for i in range(len(images)):
        image = images[i] + avg
        if to255 == True :
            image = image * 255.0
        image = np.clip(image, 0, 255).astype('uint8')
        full_path = conf.project_path + conf.output_images + path
        if len(images) != 1:
            full_path = conf.project_path + conf.output_images + path + '\\' + str(i)
        make_sure_path_exists(full_path)
        scipy.misc.imsave(full_path + name + '_' + str(i) + '.jpg', image)


def load_pictures_for_feed(directory_path, recursive=False, gen_res=304, content_res=224):
    print("Loading pictures : " + directory_path)
    images = []
    content_images = []
    for file in os.listdir(conf.project_path + conf.images_path + directory_path):
        full_path = os.path.join(conf.project_path + conf.images_path + directory_path, file)
        if os.path.isfile(full_path) and str(file)[-4:] == '.jpg' :
            img, avg_img = load_image(directory_path + '\\' + str(file), between_01=True, output_size=gen_res)

            if len(img.shape) < 3 or img.shape[2] != 3 :
                print("Picture with less than 3 channels found : " + str(file) + "\t Picture will be skipped.")
                continue

            images.append(img)
            img, avg_img = load_image(directory_path + '\\' + str(file), between_01=True, output_size=content_res)
            content_images.append(img)
        else:
            if recursive and not os.path.isfile(full_path):
                im, con = load_pictures_for_feed(directory_path + '\\' + file, recursive=recursive, gen_res=gen_res, content_res=content_res)
                for i in im:
                    images.append(i)
                for c in con:
                    content_images.append(c)

    print("Done. Number of pictures loaded : " + str(len(images)))
    return images, content_images

def tensorshape_to_int_array(ts):
    s = []
    for x in ts:
        #print(x)
        if(x.value is None):
            #print("is none")
            s.append(None)
            continue
        #print(type(x))
        s.append(int(x))
    return s
