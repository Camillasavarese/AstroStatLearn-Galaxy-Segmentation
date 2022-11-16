import numpy as np

from astropy.io import fits
from astropy.visualization import ZScaleInterval

from pycocotools import mask
from skimage import measure

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

from tqdm import tqdm

import requests
import shutil
import time
import random
import json
import os
from pathlib import Path

import torch


norm = ZScaleInterval()


import os
import numpy as np
from PIL import Image
from astropy.io import fits
import numpy
import math

from panopticapi.utils import id2rgb, rgb2id

def sky_median_sig_clip(input_arr, sig_fract, percent_fract, max_iter=100):
    """Estimating sky value for a given number of iterations

    @type input_arr: numpy array
    @param input_arr: image data array
    @type sig_fract: float
    @param sig_fract: fraction of sigma clipping
    @type percent_fract: float
    @param percent_fract: convergence fraction
    @type max_iter: max. of iterations
    @rtype: tuple
    @return: (sky value, number of iteration)

    """
    work_arr = numpy.ravel(input_arr)
    old_sky = numpy.median(work_arr)
    sig = work_arr.std()
    upper_limit = old_sky + sig_fract * sig
    lower_limit = old_sky - sig_fract * sig
    indices = numpy.where((work_arr < upper_limit) & (work_arr > lower_limit))
    work_arr = work_arr[indices]
    new_sky = numpy.median(work_arr)
    iteration = 0
    while ((math.fabs(old_sky - new_sky)/new_sky) > percent_fract) and (iteration < max_iter) :
        iteration += 1
        old_sky = new_sky
        sig = work_arr.std()
        upper_limit = old_sky + sig_fract * sig
        lower_limit = old_sky - sig_fract * sig
        indices = numpy.where((work_arr < upper_limit) & (work_arr > lower_limit))
        work_arr = work_arr[indices]
        new_sky = numpy.median(work_arr)
    return (new_sky, iteration)


def sky_mean_sig_clip(input_arr, sig_fract, percent_fract, max_iter=100):
    """Estimating sky value for a given number of iterations

    @type input_arr: numpy array
    @param input_arr: image data array
    @type sig_fract: float
    @param sig_fract: fraction of sigma clipping
    @type percent_fract: float
    @param percent_fract: convergence fraction
    @type max_iter: max. of iterations
    @rtype: tuple
    @return: (sky value, number of iteration)

    """
    work_arr = numpy.ravel(input_arr)
    old_sky = numpy.mean(work_arr)
    sig = work_arr.std()
    upper_limit = old_sky + sig_fract * sig
    lower_limit = old_sky - sig_fract * sig
    indices = numpy.where((work_arr < upper_limit) & (work_arr > lower_limit))
    work_arr = work_arr[indices]
    new_sky = numpy.mean(work_arr)
    iteration = 0
    while ((math.fabs(old_sky - new_sky)/new_sky) > percent_fract) and (iteration < max_iter) :
        iteration += 1
        old_sky = new_sky
        sig = work_arr.std()
        upper_limit = old_sky + sig_fract * sig
        lower_limit = old_sky - sig_fract * sig
        indices = numpy.where((work_arr < upper_limit) & (work_arr > lower_limit))
        work_arr = work_arr[indices]
        new_sky = numpy.mean(work_arr)
    return (new_sky, iteration)



def linear(inputArray, scale_min=None, scale_max=None):
    """Performs linear scaling of the input numpy array.

    @type inputArray: numpy array
    @param inputArray: image data array
    @type scale_min: float
    @param scale_min: minimum data value
    @type scale_max: float
    @param scale_max: maximum data value
    @rtype: numpy array
    @return: image data array
    
    """        
    #print("img_scale : linear")
    imageData=numpy.array(inputArray, copy=True)
    
    if scale_min == None:
        scale_min = imageData.min()
    if scale_max == None:
        scale_max = imageData.max()

    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = (imageData -scale_min) / (scale_max - scale_min)
    indices = numpy.where(imageData < 0)
    imageData[indices] = 0.0
    indices = numpy.where(imageData > 1)
    imageData[indices] = 1.0
    
    return imageData


def sqrt(inputArray, scale_min=None, scale_max=None):
    """Performs sqrt scaling of the input numpy array.

    @type inputArray: numpy array
    @param inputArray: image data array
    @type scale_min: float
    @param scale_min: minimum data value
    @type scale_max: float
    @param scale_max: maximum data value
    @rtype: numpy array
    @return: image data array
    
    """        
    
    #print("img_scale : sqrt")
    imageData=numpy.array(inputArray, copy=True)
    
    if scale_min == None:
        scale_min = imageData.min()
    if scale_max == None:
        scale_max = imageData.max()

    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = imageData - scale_min
    indices = numpy.where(imageData < 0)
    imageData[indices] = 0.0
    imageData = numpy.sqrt(imageData)
    imageData = imageData / math.sqrt(scale_max - scale_min)
    
    return imageData


def log(inputArray, scale_min=None, scale_max=None):
    """Performs log10 scaling of the input numpy array.

    @type inputArray: numpy array
    @param inputArray: image data array
    @type scale_min: float
    @param scale_min: minimum data value
    @type scale_max: float
    @param scale_max: maximum data value
    @rtype: numpy array
    @return: image data array
    
    """        
    
    #print("img_scale : log")
    imageData=numpy.array(inputArray, copy=True)
    
    if scale_min == None:
        scale_min = imageData.min()
    if scale_max == None:
        scale_max = imageData.max()
    factor = math.log10(scale_max - scale_min)
    indices0 = numpy.where(imageData < scale_min)
    indices1 = numpy.where((imageData >= scale_min) & (imageData <= scale_max))
    indices2 = numpy.where(imageData > scale_max)
    imageData[indices0] = 0.0
    imageData[indices2] = 1.0
    try :
        imageData[indices1] = numpy.log10(imageData[indices1])/factor
    except :
        print("Error on math.log10 for ",(imageData[i][j] - scale_min))

    return imageData


def asinh(inputArray, scale_min=None, scale_max=None, non_linear=2.0):
    """Performs asinh scaling of the input numpy array.

    @type inputArray: numpy array
    @param inputArray: image data array
    @type scale_min: float
    @param scale_min: minimum data value
    @type scale_max: float
    @param scale_max: maximum data value
    @type non_linear: float
    @param non_linear: non-linearity factor
    @rtype: numpy array
    @return: image data array
    
    """        
    
    print("img_scale : asinh")
    imageData=numpy.array(inputArray, copy=True)
    
    if scale_min == None:
        scale_min = imageData.min()
    if scale_max == None:
        scale_max = imageData.max()
    factor = numpy.arcsinh((scale_max - scale_min)/non_linear)
    indices0 = numpy.where(imageData < scale_min)
    indices1 = numpy.where((imageData >= scale_min) & (imageData <= scale_max))
    indices2 = numpy.where(imageData > scale_max)
    imageData[indices0] = 0.0
    imageData[indices2] = 1.0
    imageData[indices1] = numpy.arcsinh((imageData[indices1] - \
    scale_min)/non_linear)/factor

    return imageData

def fits_to_img(image_data):
    if len(image_data.shape) == 2:
        sum_image = image_data
    else:
        sum_image = image_data[0] - image_data[0]
        for single_image_data in image_data:
            sum_image += single_image_data  

    sum_image = sqrt(sum_image, scale_min=0, scale_max=np.amax(image_data))
    sum_image = sum_image * 200
    im = Image.fromarray(sum_image)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    return im
    #im.save(dir_path+"\\image\\"+filename+".jpg")
    #im.close()
    
def mask_to_png(mask_data):
    color = id2rgb(mask_data)
    im = Image.fromarray(np.array(color))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    return im

def generate_dataset(input_dir='MC_vis', output_dir='astrostatds',dim=101, step_size=None, preprocess=True):
    img = fits.getdata(input_dir+"/img.fits")
    rms = fits.getdata(input_dir+"/rms.fits")
    mask = fits.getdata(input_dir+"/true.fits")

    if step_size is None:
        step_size = dim 

    os.mkdir(output_dir)
    os.mkdir(output_dir+"/img")
    #os.mkdir(output_dir+"/rms")
    os.mkdir(output_dir+"/mask")
    
    count = 1
    for i in range(0, img.shape[0] - dim, step_size):
        for j in range(0, img.shape[1] - dim, step_size):
            img_temp = img[i:i+dim, j:j+dim]
            rms_temp = rms[i:i+dim, j:j+dim]
            mask_temp = mask[i:i+dim, j:j+dim]

            # Preprocess
            if preprocess:
                img_temp = fits_to_img(img_temp)     
                mask_temp = mask_to_png(mask_temp)
            
            
            #hdu = fits.PrimaryHDU(img_temp)
            #hdu.writeto(output_dir+'/img/'+ str(count).zfill(6) +'.fits')
            #hdu = fits.PrimaryHDU(rms_temp)
            #hdu.writeto(output_dir+'/rms/'+ str(count).zfill(6) +'.fits')
            #hdu = fits.PrimaryHDU(mask_temp)
            #hdu.writeto(output_dir+'/mask/'+ str(count).zfill(6) +'.fits')
            
            img_temp.save(output_dir+'/img/'+ str(count).zfill(6) +".jpg")
            img_temp.close()
            mask_temp.save(output_dir+'/mask/'+ str(count).zfill(6) +".png")
            mask_temp.close()

            count += 1
            
def annotation(gt_mask, idx):
    fortran_ground_truth_binary_mask = np.asfortranarray(gt_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contour = measure.find_contours(gt_mask, 0.5)

    # https://github.com/cocodataset/cocoapi/issues/139
    if len(contour[0].ravel().tolist()) > 4 or len(contour[0].ravel().tolist()) < 4:
        segmentation = contour[0].ravel().tolist()
    else:
        segmentation = contour[0].ravel().tolist()[:2] + contour[0].ravel().tolist()
        segmentation[3] += 1
    
    # annotations: [{segments_info:[id:, category_id:, iscrowd:0, bbox:[x1,x2,y1,y2], aerea:], file_name:(la maschera), image_id:}]
    annot = {
            "id": int(idx),
            "category_id": 1,
            "iscrowd": 0,
            "bbox": ground_truth_bounding_box.tolist(),
            "area": ground_truth_area.tolist(),
        }
    return annot

def image(img_file, height=-1, width=-1):
    img = {
            "file_name": img_file,
            #"rms_file_name": os.path.join('rms', img_file),
            "height": height,
            "width": width,
            "id": int(img_file[:-5])
        }
    return img


def generate_json(list_images, images_path, masks_path):

    images = []
    annotations = []
    for img_name in list_images:
        image_path = os.path.join(images_path, img_name)
        #img = fits.getdata(image_path)
        image_idx = int(img_name[:-5])
        mask_path = os.path.join(masks_path, img_name.replace('jpg', 'png'))
        mask = np.asarray(Image.open(mask_path), dtype=np.uint32)#fits.getdata(mask_path)
        mask = rgb2id(mask)
        n,m = mask.astype(np.float32).shape
        images.append(image(img_name, height=n, width=m))

        elements = np.delete(np.unique(mask), 0)
        segments_info = []
        for element in elements:
            gt_mask = mask == element
            segments_info.append(annotation(gt_mask, element))
        annotations.append({'segments_info': segments_info, 'file_name': img_name.replace('jpg', 'png'), 'image_id':int(img_name[:-5])})

    jsonf = {'info':{},
             'licences':{},
             'images': images,
             'annotations': annotations,
             'categories': [{"supercategory": "galaxy",
                             "id": 1,
                             "name": "star"}]
             }

    return jsonf
    

def generate_annotations(preprocess_path, test_size=0.2, val_size=0.2, seed=1234):
    
    os.mkdir(preprocess_path+"/images")
    os.mkdir(preprocess_path+"/images/train")
    os.mkdir(preprocess_path+"/images/test")
    os.mkdir(preprocess_path+"/images/val")
    os.mkdir(preprocess_path+"/annotations")
    os.mkdir(preprocess_path+"/annotations/panoptic_train")
    os.mkdir(preprocess_path+"/annotations/panoptic_test")
    os.mkdir(preprocess_path+"/annotations/panoptic_val")
    
    
    random.seed(seed)

    images_path = os.path.join(preprocess_path, 'img')
    #rms_path = os.path.join(preprocess_path, 'rms')
    gt_path = os.path.join(preprocess_path, 'mask')

    # All the elements in the images folders
    lst = os.listdir(images_path)

    random.shuffle(lst)
    len_lst = len(lst)

    test_size = int(len_lst * test_size)
    val_size = int(len_lst * val_size)

    test_images = lst[:test_size]
    val_images = lst[test_size:test_size+val_size]
    train_images = lst[test_size+val_size:]
    
    for x in test_images:
        shutil.copyfile(os.path.join(preprocess_path, 'img', x), os.path.join(preprocess_path, 'images','test', x))
        shutil.copyfile(os.path.join(preprocess_path, 'mask', x.replace('jpg', 'png')), os.path.join(preprocess_path, 'annotations','panoptic_test', x.replace('jpg', 'png')))
    for x in val_images:
        shutil.copyfile(os.path.join(preprocess_path, 'img', x), os.path.join(preprocess_path, 'images','val', x))
        shutil.copyfile(os.path.join(preprocess_path, 'mask', x.replace('jpg', 'png')), os.path.join(preprocess_path, 'annotations','panoptic_val', x.replace('jpg', 'png')))
    for x in train_images:
        shutil.copyfile(os.path.join(preprocess_path, 'img', x), os.path.join(preprocess_path, 'images','train', x))
        shutil.copyfile(os.path.join(preprocess_path, 'mask', x.replace('jpg', 'png')), os.path.join(preprocess_path, 'annotations','panoptic_train', x.replace('jpg', 'png')))

    
    jsonf_train = generate_json(train_images, images_path, gt_path) 
    jsonf_val = generate_json(val_images, images_path, gt_path)
    jsonf_test = generate_json(test_images, images_path, gt_path)
    
    shutil.rmtree(os.path.join(preprocess_path, 'img'))
    shutil.rmtree(os.path.join(preprocess_path, 'mask'))

    with open(os.path.join(preprocess_path,'annotations','panoptic_train.json'), 'w') as js_file:
        json.dump(jsonf_train, js_file)
    with open(os.path.join(preprocess_path,'annotations','panoptic_val.json'), 'w') as js_file:
        json.dump(jsonf_val, js_file)
    with open(os.path.join(preprocess_path,'annotations','panoptic_test.json'), 'w') as js_file:
        json.dump(jsonf_test, js_file)
        

def generate_dataset_with_annotations(input_dir='MC_vis', 
                                      output_dir='astrostatds',
                                      dim=101, 
                                      step_size=None, 
                                      test_size=0.2, 
                                      val_size=0.2, 
                                      seed=1234):
    generate_dataset(input_dir=input_dir, output_dir=output_dir,dim=dim, step_size=step_size)
    generate_annotations(output_dir, test_size=test_size, val_size=val_size, seed=seed)
    
    
if __name__ == "__main__":
    shutil.rmtree(os.path.join('.','astrostatds'))
    generate_dataset_with_annotations(input_dir='MC_vis', 
                                      output_dir='astrostatds',
                                      dim=256, 
                                      step_size=None, 
                                      test_size=0.2, 
                                      val_size=0.2, 
                                      seed=1234)