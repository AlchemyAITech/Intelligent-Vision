# SAM 后端调用
# （1）模型加载
# （2）根据前端返回的坐标，以及原图。获取切块图片，计算encoder结果 并返回npy文件
import cv2
import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)
from segment_anything import sam_model_registry, SamPredictor
from mpAI.common.utils.image_opt import MpImage
from mpAI.forward.base.interface import ModelBase
import base64
from sam2.build_sam import build_sam2
from sam2.sam2_predictor_all import SAM2Predictor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import torch.nn.functional as F
from mpAI.common import *


class Forward_SAM(ModelBase):
    def init(self):
        sam = sam_model_registry["vit_b"](checkpoint=self.path_dict['model.pth'])
        sam = sam.to(self.device)
        self.predictor = SamPredictor(sam)
        self.quantization = True

    def get_npy(self, image, xywh):
        '''
        Args:
            image: str, imgpath
            xywh: tuple, roi
        Returns:
            image_embedding: ndarray, of uint8, (1,256,64,64)
            max_: ndarray, of float32, (1,256,1,1)
            min_: ndarray, of float32, (1,256,1,1)
        '''
        x, y, w, h = xywh
        scale = 1024 / max(w, h)
        with MpImage(image) as mpimg:
            img = mpimg.read_rect((x, y, x + w, y + h), scale=scale)
        self.predictor.set_image(img)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        max_ = 0
        min_ = 0
        if self.quantization:
            # image_embedding = ((image_embedding + 1) * 127).astype(np.uint8)
            max_ = image_embedding.max(axis=(2, 3), keepdims=True)
            min_ = image_embedding.min(axis=(2, 3), keepdims=True)
            image_embedding = ((image_embedding - min_) /(max_-min_+0.00001)* 255).astype(np.uint8)
        
        return base64.b64encode(image_embedding).decode(), \
               base64.b64encode(np.float32(max_-min_+0.00001)).decode(), \
               base64.b64encode(np.float32(min_)).decode()

    def release(self):
        pass


    
class Forward_SAM2(ModelBase):
    # TODO 1.模型路径及调试cfg
    def init(self):
        sam2_checkpoint = self.path_dict['sam2_hiera_base_plus.pt']
        model_cfg = "sam2_hiera_b+.yaml"
        # 1：加载后端模型
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device =self.device, apply_postprocessing=False)

        self.fwd = SAM2Predictor(model=sam2,
                                 points_per_side=32,
                                 points_per_batch=64,
                                 pred_iou_thresh=0.7,
                                 stability_score_thresh=0.90,
                                 stability_score_offset=0.7,
                                 box_nms_thresh=0.4)
        self.K_Neighbors = 5
        self.score_threshold = 0.2
        self.iou_threshold = 0.5 #去重用
        self.quantization = True


    def get_mask_all(self, image):
        '''
        Args:
            image: ndarray
        Returns:
            masks
            mask_feature
        '''
        masks, mask_feature = self.fwd.generate_step2(image)
        return masks, mask_feature
    

    def get_image_embedding(self, image, xywh):
        '''
        Args:
            image: str, imgpath
            xywh: tuple, roi
        Returns:
            image_embedding: ndarray, of uint8, (1,256,64,64)            
            max_: ndarray, of float32, (1,256,1,1)
            min_: ndarray, of float32, (1,256,1,1)
        '''
        x, y, w, h = xywh
        scale = 1024 / max(w, h)
        with MpImage(image) as mpimg:
            img = mpimg.read_rect((x, y, x + w, y + h), scale=scale)

        self.fwd.predictor.set_image(img)
        image_embedding = self.fwd.predictor.get_image_embedding().cpu().numpy()
        max_ = 0
        min_ = 0
        if self.quantization:
            # image_embedding = ((image_embedding + 1) * 127).astype(np.uint8)
            max_ = image_embedding.max(axis=(2, 3), keepdims=True)
            min_ = image_embedding.min(axis=(2, 3), keepdims=True)
            image_embedding = ((image_embedding - min_) /(max_-min_+0.00001)* 255).astype(np.uint8)
        
        return base64.b64encode(image_embedding).decode(), \
               base64.b64encode(np.float32(max_-min_+0.00001)).decode(), \
               base64.b64encode(np.float32(min_)).decode()
    
    # init AI
    def get_image_feature(self, image, xywh):
        '''
        Args:
            image: str, imgpath
            xywh: tuple, roi
        Returns:
            image_embedding: ndarray, of uint8, (1, 256, 64, 64)
            high_res_feats0: ndarray, of uint8, (1, 64, 128, 128)
            high_res_feats1: ndarray, of uint8, (1, 32, 256, 256)
            max_: ndarray, of float32, (1,256+64+32,1,1)
            min_: ndarray, of float32, (1,256+64+32,1,1)
        '''
        x, y, w, h = xywh
        scale = 1024 / max(w, h)
        with MpImage(image) as mpimg:
            img = mpimg.read_rect((x, y, x + w, y + h), scale=scale)
        self.fwd.predictor.set_image(img)
        image_embedding, high_res_feats = self.fwd.predictor.get_image_feature()
        image_embedding = image_embedding.cpu().numpy()
        high_res_feats0, high_res_feats1 = high_res_feats[0].cpu().numpy(), high_res_feats[1].cpu().numpy()
    
        max_image_embedding = 0
        min_image_embedding = 0
        max_high_res_feats0 = 0
        min_high_res_feats0 = 0
        max_high_res_feats1 = 0
        min_high_res_feats1 = 0
        
        if self.quantization:
            # 计算每个特征图的最大值和最小值
            max_image_embedding = image_embedding.max(axis=(2, 3), keepdims=True)
            min_image_embedding = image_embedding.min(axis=(2, 3), keepdims=True)
            
            max_high_res_feats0 = high_res_feats0.max(axis=(2, 3), keepdims=True)
            min_high_res_feats0 = high_res_feats0.min(axis=(2, 3), keepdims=True)
            
            max_high_res_feats1 = high_res_feats1.max(axis=(2, 3), keepdims=True)
            min_high_res_feats1 = high_res_feats1.min(axis=(2, 3), keepdims=True)
            
            # 对每个特征图进行归一化和量化
            image_embedding = ((image_embedding - min_image_embedding) / 
                            (max_image_embedding - min_image_embedding + 0.00001) * 255).astype(np.uint8)
            
            high_res_feats0 = ((high_res_feats0 - min_high_res_feats0) / 
                            (max_high_res_feats0 - min_high_res_feats0 + 0.00001) * 255).astype(np.uint8)
            
            high_res_feats1 = ((high_res_feats1 - min_high_res_feats1) / 
                            (max_high_res_feats1 - min_high_res_feats1 + 0.00001) * 255).astype(np.uint8)
        
            # 合并 max_ 和 min_，以适应返回的大小
            max_combined = np.concatenate([max_image_embedding, max_high_res_feats0, max_high_res_feats1], axis=1)
            min_combined = np.concatenate([min_image_embedding, min_high_res_feats0, min_high_res_feats1], axis=1)

                # 确保数组是C连续的，然后再编码
            image_embedding_contiguous = np.ascontiguousarray(image_embedding)
            high_res_feats0_contiguous = np.ascontiguousarray(high_res_feats0)
            high_res_feats1_contiguous = np.ascontiguousarray(high_res_feats1)
            max_combined_contiguous = np.ascontiguousarray(max_combined)
            min_combined_contiguous = np.ascontiguousarray(min_combined)
        
        return base64.b64encode(image_embedding_contiguous).decode(),  \
            base64.b64encode(high_res_feats0_contiguous).decode(), \
            base64.b64encode(high_res_feats1_contiguous).decode(), \
            base64.b64encode(np.float32(max_combined_contiguous)).decode(), \
            base64.b64encode(np.float32(min_combined_contiguous)).decode()
    
    def get_masks_feature(self, masks, feature):
        '''
        Args:
            masks: ndarray, of int, (n, h, w)
            feature: ndarray, of float, (1, c, h, w)
        Returns:
            mask_feature: ndarray, of float, (n, c)'''

        low_res_masks = torch.tensor(masks).float().to(feature.device)[:, None, :, :] * 60 - 30
        low_res_masks = F.interpolate(low_res_masks, size=feature.shape[-2:], mode='bilinear', align_corners=False)

        feature = feature.flatten(2, 3)

        low_res_masks = low_res_masks.flatten(2, 3)
        masks_low_res = (low_res_masks > 0).float()
        topk_idx = torch.topk(low_res_masks, 1)[1]

        masks_low_res.scatter_(2, topk_idx, 1.0)

        mask_feature = (feature * masks_low_res).sum(dim=2) / masks_low_res.sum(dim=2)
        del low_res_masks, masks_low_res
        # mask_feature = F.normalize(mask_feature, dim=1)
        return mask_feature.cpu().numpy()

    def _points2mask(self, inp_shapes:List[M_shape], image):
        '''
        Args:
            image: ndarray
        Returns:
            mask: ndarray, of int, (h, w)
        '''
        # scale = 256/max(image.shape[:2])
        # points = np.array(points) * scale
        scale_x = 256 / image.shape[1]
        scale_y = 256 / image.shape[0]
        masks = np.zeros((len(inp_shapes), 256, 256), dtype=np.uint8)
        for i, sp in enumerate(inp_shapes):
            deepcopy(sp).scale([scale_x, scale_y]).draw_on(masks[i])
        return masks
    
    def _mask2points(self, masks):
        '''
        Args:
            masks: ndarray, of int, (n, h, w)
        Returns:
            points: list(ndarray), of float, [(k, 2), ...], k is the number of points in each contour
        '''
        points = []
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points.append(contours.reshape(-1, 2))
        return points

    def _calculate(self, feature_in, label_in, feature_all, masks_all, k=5, score_threshold=0.5):
        # 计算训练特征和测试特征之间的余弦相似度
        similarities = cosine_similarity(feature_all, feature_in)
        
        # 获取每个测试样本的K个最近邻
        sorted_indices = np.argsort(-similarities, axis=1)[:, :k]
        sorted_similarities = -np.sort(-similarities, axis=1)[:, :k]
        
        res = []
        # 遍历所有测试样本
        for i in range(feature_all.shape[0]):
            class_votes = defaultdict(float)
            class_num = defaultdict(int)
            max_sim = 0
            
            # 对K个最近邻进行加权投票
            for j in range(min(k, len(feature_in))):
                index = sorted_indices[i, j]
                similarity = sorted_similarities[i, j]
                class_label = label_in[index]
                class_votes[class_label] += similarity
                class_num[class_label] += 1
                if similarity > max_sim:
                    max_sim = similarity
            for key, score in class_votes.items():
                class_votes[key] = score / class_num[key]
            
            # 找到得票最多的类别
            predicted_label = max(class_votes, key=class_votes.get)
            max_score = class_votes[predicted_label] #/ class_num[predicted_label]

            # 过滤分数过低的结果
            if max_score >= score_threshold:
                contours, _ = cv2.findContours(masks_all[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    res.append(M_polygon(
                        predicted_label, largest_contour.reshape(-1,2), max_score))
        return res
    
    # TODO 前端返回的轮廓，结合mask all的结果进行匹配, 并返回匹配好的轮廓结果
    # init all
    def select_mask(self, inp_shapes, image, xywh) -> List[Dict]:
        '''
        Args:
            inp_shapes: list(dict), 已标的轮廓，（0,0）点为图像原点
            image: str, imgpath
            xywh: tuple, roi
        '''
        # 如果points为空(长度为0)，则直接返回空
        if len(inp_shapes) == 0:
            return []

        x, y, w, h = xywh
        scale = 1024 / max(w, h)
        with MpImage(image) as mpimg:
            img = mpimg.read_rect((x, y, x + w, y + h), scale=scale)
            inp_shapes = [sp.shift_xy(-x, -y).scale(scale)
                          for sp in parse_shape_list(inp_shapes)]

        masks = self._points2mask(inp_shapes, img)

        # init image feature
        self.fwd.predictor.set_image(img)
        feature = self.fwd.predictor.get_image_embedding()

        # # get mask feature
        mask_feature = self.get_masks_feature(masks, feature)

        # get mask all + feature (去掉内部函数中的 set_image(image))
        masks_all, mask_feature_all = self.get_mask_all(img)

        # TODO K-means select mask and score
        res = self._calculate(mask_feature, [sp.label for sp in inp_shapes],
                              mask_feature_all, masks_all, k=self.K_Neighbors, score_threshold=self.score_threshold)

        # 去重
        ioumat = ShapeAnalysis.calc_iou_mat(res, inp_shapes)
        oup = []
        for i in range(len(res)):
            if np.max(ioumat[i]) < self.iou_threshold: # 0.5
                oup.append(res[i].scale(1/scale).shift_xy(x,y).to_dict())

        return oup

    def release(self):
        pass
    