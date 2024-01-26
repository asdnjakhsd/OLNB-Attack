# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import copy

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from utils import permutation_utils


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
                    help='datasets')
parser.add_argument('--config', default='', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    total_vot = 0
    mean_vot_path = os.path.join('results', args.dataset, "SiamRPN++(Original)", 'baseline', 'mean_vot.txt')

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        flag = 0
        iter_count = 0
        # restart tracking
        for v_idx, video in enumerate(dataset):
            #if video.name not in ['crabs1']:
            #   continue
            
            #if flag or video.name == "ants1":
            #    flag = 1
            #else:
            #    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            total_time = 0
            pred_bboxes = []

            perturb_max = 10000
            batch_size = 1
            num_channels = 3
            rho = 12
            eps = 9
            freq_dims = 38
            stride = 9														

            square_size = -1
            target_x, target_y = -1, -1

            print("&&&&&& processing: ", video.name)
            
            scores_sum = 0
            scores_num = 0

            seg_x = 0
            seg_y = 0
            l2_normes = []
            square_size = 299
            size_group = (square_size, square_size)
            for idx, (img, gt_bbox) in enumerate(video):
              
                print("--------- processing frame no. ", idx)

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                               gt_bbox[0]+gt_bbox[2] -
                               1, gt_bbox[1]+gt_bbox[3]-1,
                               gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]

               

                if idx == 0:
                    last_preturb = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                    orgin_square_size = int(permutation_utils.get_square_size(img))
                    n_dims = num_channels * square_size * square_size
                    epsi = eps * torch.ones(batch_size)

                    seg_x = cx
                    seg_y = cy

                if idx == frame_counter:
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    save_bbox = gt_bbox_
                elif idx > frame_counter:
                    image = img
                    origin_tensor = permutation_utils.cvmat_to_tensor(image)
                    
                    outputs_orig = tracker.track_fixed(image)
                    
                    last_gt = save_bbox 

                    heavy_noise = np.random.randint(-1, 1, (image.shape[0], image.shape[1], image.shape[2])) * 64
                    image_noise = image + heavy_noise
                    image_noise = np.clip(image_noise, 0, 255).astype(np.uint8)
      
                    outputs_target = tracker.track_fixed(image_noise)
                    target_score = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_target['bbox']))
                        
                    cv_initial_img_float32 = image.astype(np.float32)
                    cv_adv_image = image
                    

                    delta = 3
                    weight = 0
                    
                    origin_size_group = (orgin_square_size, orgin_square_size)
                    
                    target_x, target_y = permutation_utils.get_square_pos(origin_tensor, int(seg_x + orgin_square_size//2), int(seg_x + orgin_square_size//2), orgin_square_size)
                    

                    ts_pure_square = permutation_utils.cvmat_to_tensor(cv2.resize(permutation_utils.get_square_cvmat(image, target_x, target_y, orgin_square_size), size_group))
                    
                    # preapreation
                    k = 0
                    step = 25
                    iter_limit = 2
                    scores_arr = []
                    ts_arr = []
                    dims_record = []
                    sign_record = []
                    distortion = ts_pure_square - ts_pure_square
                    square_k = ts_pure_square
                    indices = permutation_utils.block_order(square_size, 3, freq_dims, stride)
                    stop_flag = 0
                    n_steps = 0
                    while True:
                        scores_arr = []
                        ts_arr = []
                        
                        #indices = permutation_utils.block_order(square_size, 3, freq_dims, stride)
                        found = 0
                        i = 0
                        n_steps += 1
                        while i < step:
                            # trying left direction
                            
                            dim = indices[k % n_dims]
                            i += 1
                            k += 1
                            diffL = torch.zeros(batch_size, n_dims)
                            diffL[:, dim] = -epsi   # left perturbation
                            
                            
                            cur_perturb = permutation_utils.l2_projection(
                                distortion, permutation_utils.block_idct(permutation_utils.expand_vector(diffL, square_size, square_size), square_size, linf_bound=0.0), rho)  
                            square_left = (square_k + delta * cur_perturb).clamp(0, 1)
                            cv_left = permutation_utils.set_cvsquare_to_img(image, target_x, target_y, cv2.resize(permutation_utils.tensor_to_cvmat(square_left), origin_size_group))
                            outputs_adv = tracker.track_fixed(cv_left.astype(np.uint8))

                            score_left = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))
                            
                            if (score_left < target_score):
                                scores_arr.append(score_left)
                                ts_arr.append(square_left)
                                dims_record.append(dim)
                                sign_record.append(-1)
                                found = 1
                                continue
                            
                            diffR = torch.zeros(batch_size, n_dims)
                            diffR[:, dim] = epsi    # right perturbation

                            # trying right direction
                            cur_perturb = permutation_utils.l2_projection(
                                distortion, permutation_utils.block_idct(permutation_utils.expand_vector(diffR, square_size, square_size), square_size, linf_bound=0.0), rho)
                            square_right = (square_k + delta * cur_perturb).clamp(0, 1)
                            
                            cv_right = permutation_utils.set_cvsquare_to_img(image, target_x, target_y, cv2.resize(permutation_utils.tensor_to_cvmat(square_right), origin_size_group))
                            
                            outputs_adv = tracker.track_fixed(cv_right.astype(np.uint8))
                            score_right = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))

                            
                            if (score_right < target_score):
                                scores_arr.append(score_right)
                                ts_arr.append(square_right)
                                dims_record.append(dim)
                                sign_record.append(1)
                                found = 1
                                k += 1
                    
                        if len(scores_arr)  > 0:
                            min_score_index = np.argmin(scores_arr)
                            square_k = ts_arr[min_score_index]
                    
                            cv_square = cv2.resize(permutation_utils.tensor_to_cvmat(square_k), origin_size_group)
                            cv_adv_image = permutation_utils.set_cvsquare_to_img(image, target_x, target_y, cv_square)
                            outputs_adv = tracker.track_fixed(cv_adv_image.astype(np.uint8))
                            temp_score = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))
                    
                            if temp_score == 0 :
                                print("get zero scores.")
                                stop_flag = 1
                        
                        
                        scores_arr = []
                        ts_arr = []
                        dims_record = []
                        sign_record = []
                    
                        l2_norm = np.mean(get_diff(cv_initial_img_float32, cv_adv_image.astype(np.float32)))
                        if l2_norm > perturb_max or n_steps >= iter_limit or stop_flag:
#                           iter_count += k
                            stop_flag = 0
                            if (found == 0):
                                cv_adv_image = image_noise
                            l2_normes.append(l2_norm)
                            break
                    last_preturb = cv_adv_image - image
                    img = cv_adv_image

                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    [resx, resy, resw, resh] = pred_bbox
                    seg_x = resx
                    seg_y = resy
                    
                    #cv2.rectangle(img, pt1 = (int(pred_x), int(pred_y)), pt2 = (int(pred_x+pred_w),int(pred_y+pred_h)), color = (0,0,255), thickness = 2)
                    #cv2.rectangle(img, pt1 = (int(cx), int(cy)), pt2 = (int(cx+w),int(cy+h)), color = (255,0,0), thickness = 2)
                    #cv2.rectangle(img, pt1 = (int(target_x), int(target_y)), pt2 = (int(target_x+orgin_square_size),int(target_y+orgin_square_size)), color = (0,255,0),thickness = 2)
                    #cv2.imwrite('./__pic/pic'+str(idx)+'.jpg', img)
                    #cv2.waitKey()

                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        scores_sum += overlap
                        scores_num += 1
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                total_time += toc
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                      True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            model_name = 'SiamRPN++(Original)'
            # save results
            
            video_path = os.path.join('results', args.dataset, model_name,'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(
                video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            means = scores_sum / scores_num
            with open(mean_vot_path, 'a') as f:
                 f.write(video.name + ":  " + str(means) + "\n")
            total_vot += means
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        with open(mean_vot_path, 'a') as f:
            f.write("total :  " + str(total_vot / 60) + "\n")
    else:
        # OPE tracking
        flag = 0
        for v_idx, video in enumerate(dataset):
            # if video.name not in ['car1']:
            #     continue
            
            #if flag or video.name == "Suv":
            #     flag = 1
            #else:
            #     continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []

            perturb_max = 10000
            batch_size = 1
            num_channels = 3
            rho = 12
            eps = 9
            freq_dims = 38
            stride = 9																							

            square_size = -1
            target_x, target_y = -1, -1

            print("&&&&&& processing: ", video.name)
            
            scores_sum = 0
            scores_num = 0

            seg_x = 0
            seg_y = 0
            l2_normes = []
            square_size = 299
            size_group = (square_size, square_size)
            
            for idx, (img, gt_bbox) in enumerate(video):
                print("--------- processing frame no. ", idx)

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                               gt_bbox[0]+gt_bbox[2] -1, gt_bbox[1]+gt_bbox[3]-1,
                               gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]



                if idx == 0:
                  
                    tracker.init(img, gt_bbox_)

                    last_preturb = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                    pred_bbox = gt_bbox_
                    
                    pred_bboxes.append(pred_bbox)
                    save_bbox = gt_bbox_
                    
                    orgin_square_size = int(permutation_utils.get_square_size(img))
                    n_dims = num_channels * square_size * square_size
                    epsi = eps * torch.ones(batch_size)

                    seg_x = cx
                    seg_y = cy

                else:
                    image = img
                    origin_tensor = permutation_utils.cvmat_to_tensor(image)
                    
                    outputs_orig = tracker.track_fixed(image)
                    
                    last_gt = save_bbox 

                    heavy_noise = np.random.randint(-1, 1, (image.shape[0], image.shape[1], image.shape[2])) * 64
                    image_noise = image + heavy_noise
                    image_noise = np.clip(image_noise, 0, 255).astype(np.uint8)
      
                    outputs_target = tracker.track_fixed(image_noise)
                    target_score = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_target['bbox']))
                        
                    cv_initial_img_float32 = image.astype(np.float32)
                    cv_adv_image = image
                    

                    delta = 7
                    weight = 0
                    
                    origin_size_group = (orgin_square_size, orgin_square_size)
                    
                    target_x, target_y = permutation_utils.get_square_pos(origin_tensor, int(seg_x + orgin_square_size//2), int(seg_x + orgin_square_size//2), orgin_square_size)
                    
                    cv_initial_sample = np.clip(cv_initial_img_float32 + weight * last_preturb, 0, 255).astype(np.uint8)
                    ts_initial_square = permutation_utils.cvmat_to_tensor(cv2.resize(permutation_utils.get_square_cvmat(cv_initial_sample, target_x, target_y, orgin_square_size), size_group))
                    outputs_adv = tracker.track_fixed(cv_initial_sample.astype(np.uint8))
                
                    threshold = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))

                    ts_pure_square = permutation_utils.cvmat_to_tensor(cv2.resize(permutation_utils.get_square_cvmat(image, target_x, target_y, orgin_square_size), size_group))
                    
                    square_k = ts_pure_square
                    # preapreation
                    k = 0
                    step = 100
                    iter_limit = 2
                    scores_arr = []
                    ts_arr = []
                    dims_record = []
                    sign_record = []
                    distortion = square_k - ts_pure_square
                    square_k = ts_initial_square
                    indices = permutation_utils.block_order(square_size, 3, freq_dims, stride)
                    stop_flag = 0
                    n_steps = 0
                    while True:
                        scores_arr = []
                        ts_arr = []
                        
                        found = 0
                        i = 0
                        n_steps += 1
                        while i < step:
                            
                            # trying left direction
                            
                            dim = indices[k % n_dims]
                            i += 1
                            k += 1
                            diffL = torch.zeros(batch_size, n_dims)
                            diffL[:, dim] = -epsi   # left perturbation
                            
                            
                            cur_perturb = permutation_utils.l2_projection(
                                distortion, permutation_utils.block_idct(permutation_utils.expand_vector(diffL, square_size, square_size), square_size, linf_bound=0.0), rho)  
                            square_left = (square_k + delta * cur_perturb).clamp(0, 1)
                            cv_left = permutation_utils.set_cvsquare_to_img(image, target_x, target_y, cv2.resize(permutation_utils.tensor_to_cvmat(square_left), origin_size_group))
                            outputs_adv = tracker.track_fixed(cv_left.astype(np.uint8))

                            score_left = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))
                            
                            if (score_left < threshold):
                                scores_arr.append(score_left)
                                dims_record.append(dim)
                                sign_record.append(-1)
                                found = 1
                                continue
                            
                            diffR = torch.zeros(batch_size, n_dims)
                            diffR[:, dim] = epsi    # right perturbation

                            # trying right direction
                            cur_perturb = permutation_utils.l2_projection(
                                distortion, permutation_utils.block_idct(permutation_utils.expand_vector(diffR, square_size, square_size), square_size, linf_bound=0.0), rho)
                            square_right = (square_k + delta * cur_perturb).clamp(0, 1)
                            
                            cv_right = permutation_utils.set_cvsquare_to_img(image, target_x, target_y, cv2.resize(permutation_utils.tensor_to_cvmat(square_right), origin_size_group))
                            
                            outputs_adv = tracker.track_fixed(cv_right.astype(np.uint8))
                            score_right = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))

                            
                            if (score_right < threshold):
                                scores_arr.append(score_right)
                                dims_record.append(dim)
                                sign_record.append(1)
                                found = 1
                                k += 1
#                               ts_arr.append(square_right)
                    
                        if len(scores_arr)  > 0:
                            sorted_scores_index = np.argsort(scores_arr)
                            diff_final = torch.zeros(batch_size, n_dims)
                            diff_final[:, dims_record[sorted_scores_index[0]]] = epsi * sign_record[sorted_scores_index[0]]
                            final_perturb = permutation_utils.l2_projection(
                                distortion, permutation_utils.block_idct(permutation_utils.expand_vector(diff_final, square_size, square_size), square_size, linf_bound=0.0), rho)
                    
                            square_k = (square_k + delta * final_perturb).clamp(0, 1)
                    
                            cv_square = cv2.resize(permutation_utils.tensor_to_cvmat(square_k), origin_size_group)
                            cv_adv_image = permutation_utils.set_cvsquare_to_img(image, target_x, target_y, cv_square)
                            outputs_adv = tracker.track_fixed(cv_adv_image.astype(np.uint8))
                            temp_score = vot_overlap(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']), (img.shape[1], img.shape[0]))
                    
                            if temp_score == 0 :
                                stop_flag = 1
                        
                        
                        scores_arr = []
                        ts_arr = []
                        dims_record = []
                        sign_record = []
                    
                        l2_norm = np.mean(get_diff(cv_initial_img_float32, cv_adv_image.astype(np.float32)))
                        if l2_norm > perturb_max or n_steps >= iter_limit or stop_flag:
#                           iter_count += k
                            stop_flag = 0
                            if (found == 0):
                                cv_adv_image = image_noise
                            l2_normes.append(l2_norm)
                            break
                    last_preturb = cv_adv_image - image
                    img = cv_adv_image
                    save_bbox = gt_bbox_

                  
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        scores_sum += overlap
                        scores_num += 1
                    else:
                        # lost object
                        print("^^^^^^^^^lost in frame ", idx)
                        lost_number += 1
                    

                toc += cv2.getTickCount() - tic
                
                
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            model_name = 'SiamRPN++(Original)'
            # save results
            model_path = os.path.join('results', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
                    
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.transpose(rect1)

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0]+rect1[:, 2], rect2[:, 0]+rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1]+rect1[:, 3], rect2[:, 1]+rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2]*rect1[:, 3] + rect2[:, 2]*rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou
  
def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, sample_1.shape[0], sample_1.shape[1])
    sample_2 = sample_2.reshape(3, sample_2.shape[0], sample_2.shape[1])
    sample_1 = np.resize(sample_1, (3, 271, 271))
    sample_2 = np.resize(sample_2, (3, 271, 271))

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)

def temp_track_result( _tracker, _img):
    temp_tracker = copy.deepcopy(_tracker)
    return temp_tracker.track(_img)

def forward_perturbation(epsilon, prev_sample, target_sample):
  perturb = (target_sample - prev_sample).astype(np.float32)
  perturb /= get_diff(target_sample, prev_sample)
  perturb *= epsilon
  return perturb

if __name__ == '__main__':
    #torch.cuda.set_device(1)
    main()
