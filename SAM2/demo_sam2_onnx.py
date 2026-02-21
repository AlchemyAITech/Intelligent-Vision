import os
from copy import deepcopy

import numpy as np
import onnxruntime
import cv2
from CommonModels.models.SAM2.src.utils import MouseEvent, show_mask_cv
from collections import defaultdict
from mpAI.common import *
from mpAI.loader import ModelLoaderKept
from MyUtils import ltrb2xywh
import base64

class inputDataTorch():
    def __init__(self, ori_shape):
        self.reset_data()
        self.ori_shape = ori_shape

    def reset_data(self):
        self.ort_inputs = {
            "point_coords": None,
            "point_labels": None,
            "box": None,
            "mask_input": None,
            "multimask_output": False,
        }

    def add_point(self, point, label):
        input_point = np.array([point])
        input_label = np.array([label])
        if self.ort_inputs["point_labels"] is None:
            self.ort_inputs["point_coords"] = input_point
            self.ort_inputs["point_labels"] = input_label
        else:
            self.ort_inputs["point_coords"] = np.concatenate([input_point, self.ort_inputs["point_coords"]], axis=0)
            self.ort_inputs["point_labels"] = np.concatenate([input_label, self.ort_inputs["point_labels"]], axis=0)

    def add_box(self, box):
        input_box = np.array(box)
        self.ort_inputs["box"] = input_box[None, :] 
        
    
    def add_mask(self, Contour):
        h, w = self.ori_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        Contour = np.array(Contour)
        cv2.drawContours(mask, [Contour], -1, 60, -1)

        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = mask - 30
        self.ort_inputs["mask_input"] = mask.reshape(1, 256, 256).astype(np.float32)
        x_min = min(w-1, max(0, Contour[:, 0].min()))
        x_max = min(w-1, max(0, Contour[:, 0].max()))
        y_min = min(h-1, max(0, Contour[:, 1].min()))
        y_max = min(h-1, max(0, Contour[:, 1].max()))
        input_box = np.array([x_min, y_min, x_max, y_max])
        self.ort_inputs["box"] = input_box[None, :] 

    # update mask
    def update(self, mask):
        self.ort_inputs["mask_input"] = mask
    

class inputDataOnnx():
    def __init__(self, image_embedding, high_res_feats_0, high_res_feats_1, orig_im_size):
        self.image_embedding = image_embedding
        self.high_res_feats_0 = high_res_feats_0
        self.high_res_feats_1 = high_res_feats_1
        self.orig_im_size = np.array(orig_im_size, dtype=np.float32)
        self.reset_data()
    
    def reset_data(self):
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        onnx_coord = np.array([[0.0, 0.0]])[None, :, :].astype(np.float32)
        onnx_label = np.array([-1])[None, :].astype(np.float32)
        self.ort_inputs = {
            "image_embed": self.image_embedding,
            "high_res_feats_0": self.high_res_feats_0,
            "high_res_feats_1": self.high_res_feats_1,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": self.orig_im_size
        }

    def apply_coords(self, coords) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = self.orig_im_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * 1024 / old_w
        coords[..., 1] = coords[..., 1] * 1024 / old_h
        return coords

    def add_point(self, point, label):
        input_point = self.apply_coords(np.array([point]))[None, :, :].astype(np.float32)
        input_label = np.array([label])[None, :].astype(np.float32)
        
        if self.ort_inputs["point_labels"][0, 0] == 2:
            self.ort_inputs["point_coords"] = np.concatenate([self.ort_inputs["point_coords"][:,:2,:],
                                                              input_point, 
                                                              self.ort_inputs["point_coords"][:,2:,:]], axis=1)
            self.ort_inputs["point_labels"] = np.concatenate([self.ort_inputs["point_labels"][:,:2], 
                                                              input_label, 
                                                              self.ort_inputs["point_labels"][:,2:]], axis=1)
        else:
            self.ort_inputs["point_coords"] = np.concatenate([input_point, self.ort_inputs["point_coords"]], axis=1)
            self.ort_inputs["point_labels"] = np.concatenate([input_label, self.ort_inputs["point_labels"]], axis=1)
    

    def add_box(self, box):
        input_box = self.apply_coords(np.array(box).reshape(2, 2))[None, :, :].astype(np.float32)
        input_label = np.array([2,3])[None, :].astype(np.float32)
        if self.ort_inputs["point_labels"][0, 0] == 2:
            self.ort_inputs["point_coords"] = np.concatenate([input_box, self.ort_inputs["point_coords"][:,2:,:] ], axis=1)
            self.ort_inputs["point_labels"] = np.concatenate([input_label, self.ort_inputs["point_labels"][:,2:]], axis=1)
        else:
            self.ort_inputs["point_coords"] = np.concatenate([input_box, self.ort_inputs["point_coords"]], axis=1)
            self.ort_inputs["point_labels"] = np.concatenate([input_label, self.ort_inputs["point_labels"]], axis=1)
        
    
    def add_mask(self, Contour):
        h, w = self.orig_im_size
        mask = np.zeros((int(h), int(w)), dtype=np.uint8)
        Contour = np.array(Contour)
        cv2.drawContours(mask, [Contour], -1, 60, -1)

        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        mask = mask - 30
        self.ort_inputs["mask_input"] = mask.reshape(1, 1, 256, 256).astype(np.float32)
        self.ort_inputs["has_mask_input"] = np.ones(1, dtype=np.float32)
        x_min = min(w-1, max(0, Contour[:, 0].min()))
        x_max = min(w-1, max(0, Contour[:, 0].max()))
        y_min = min(h-1, max(0, Contour[:, 1].min()))
        y_max = min(h-1, max(0, Contour[:, 1].max()))
        input_box = np.array([x_min, y_min, x_max, y_max])
        self.add_box(input_box)
        # self.ort_inputs["box"] = input_box[None, :] 

    # update mask
    def update(self, mask):
        self.ort_inputs["mask_input"] = mask
        self.ort_inputs["has_mask_input"] = np.ones(1, dtype=np.float32)
    
  

def demo(model_src, img_src, xywh):
    curdir = os.path.dirname(os.path.abspath(__file__))
    fwd_type = "onnx" # # "torch"
    # fwd_type = "torch" # "onnx" # 
    # img_path

    # 1：加载后端模型
    model_B = ModelLoaderKept.load(model_src)

    with MpImage(img_src) as mpimg:
        img_crop = mpimg.read_rect(xywh2ltrb(*xywh))
    # img_crop = img.read_rect()
    cv2.namedWindow("image", 0)

    cv2.imshow('image', img_crop)
    cv2.waitKey(10)

    # 4：后端跑encoder模型，提供npy文件
    emb_npy, feat0, feat1, max_, min_ = model_B.get_image_feature(img_src, xywh)
    image_embedding = np.frombuffer(base64.b64decode(emb_npy), dtype=np.uint8).reshape((1, 256, 64, 64))
    feat0 = np.frombuffer(base64.b64decode(feat0), dtype=np.uint8).reshape((1, 32, 256, 256))
    feat1 = np.frombuffer(base64.b64decode(feat1), dtype=np.uint8).reshape((1, 64, 128, 128))
    max_ = np.frombuffer(base64.b64decode(max_), dtype=np.float32).reshape((1, 256+64+32, 1, 1))
    min_ = np.frombuffer(base64.b64decode(min_), dtype=np.float32).reshape((1, 256+64+32, 1, 1))

    image_embedding = image_embedding.astype(np.float32)/255.*(max_[:,:256,:,:]- min_[:,:256,:,:] + 0.00001) + min_[:,:256,:,:]
    feat0 = feat0.astype(np.float32)/255.*(max_[:,256:256+32,:,:]- min_[:,256:256+32,:,:] + 0.00001) + min_[:,256:256+32,:,:]
    feat1 = feat1.astype(np.float32)/255.*(max_[:,256+32:256+32+64,:,:]- min_[:,256+32:256+32+64,:,:] + 0.00001) + min_[:,256+32:256+32+64,:,:]

    # image_embedding, feat0, feat1 = model_B.get_image_feature(img_crop)
    print("init model done.")
    cv2.waitKey(10)

    # 5：前端根据npy文件给出初始mask
    # 初始化onnx模型, 初始化数据有以下两种形式：
    # （1）box的形式
    # （2）point的形式
    # （3）mask的形式
    if fwd_type == "torch":
        # torch model
        indata = inputDataTorch(ori_shape = img_crop.shape[:2])
    else:
        # onnx model
        # 
        model_path = os.path.join(curdir, 'segmodel2_q')
        # model_path = os.path.join(curdir, 'checkpoints/sam2_hiera_small_decoder.onnx')
        ort_session = onnxruntime.InferenceSession(model_path)
        indata = inputDataOnnx(image_embedding, feat0, feat1, orig_im_size = img_crop.shape[:2])

    ME = MouseEvent(img_crop, title='image')
    print("\n开始初始标注，\n b : 以box的形式标注\n n : 以point的形式标注\n 请输入：")
    while True:
        k = cv2.waitKey(1)
        # （1）box的形式
        if k == ord('b'):
            label, rect = ME("rect")
            indata.add_box(rect)
            break
        # （2）point的形式
        elif k == ord('n'):
            label, point = ME("point")
            indata.add_point(point, label)
            break
        # （3）mask的形式
        # TODO，效果不佳
        elif k == ord('m'):
            label, Contour = ME("mask")
            indata.add_mask(Contour)
            break
    
    # forward
    if fwd_type == "torch":
        masks, _, low_res_logits, mk_feature = model_B.fwd.predictor.predict(**indata.ort_inputs)
        print("_", _)
        print("low_res_logits: ", low_res_logits.shape)
        print("masks: ", masks.shape)
    else:
        masks, score, low_res_logits = ort_session.run(None, indata.ort_inputs)
        print("low_res_logits: ", low_res_logits.shape)
        print("score: ", score.shape)
        print("masks: ", masks.shape)
        masks = masks[0]
    
    masks = masks > 0
    mask_img = show_mask_cv(masks[0], img_crop, random_color=False)
    ME.img = mask_img
    indata.update(low_res_logits)

    # 6：结合mask的修正
    #（1）points输入：0：负类，去除区域；1：正类，增加区域
    #（2）box输入：转化为points，label为 2 3，起始点和终止点坐标
    #（3）重置：重置后回到初始状态 5
    #（4）退出：结束标图
    #（5）保存：保存该区域后，同时回到初始状态 5
    inputs = []
    while True:
        cv2.imshow('image', ME.img)
        print("\n开始修正标注，\n b : 以box的形式标注，\n n : 以point的形式标注，\n a : 保存当前标注，\n i : 重置标注，\n q : 结束标注")
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        # 重置
        elif k == ord('i'):
            print("\n开始重置标注，\n b : 以box的形式标注，\n n : 以point的形式标注，\n 其他 : 取消重置")
            k = cv2.waitKey(0)
            # （1）box的形式
            if k == ord('b'):
                indata.reset_data()
                label, rect = ME("rect")
                indata.add_box(rect)
            # （2）point的形式
            elif k == ord('n'):
                indata.reset_data()
                label, point = ME("point")
                indata.add_point(point, label)
            else:
                continue
            if fwd_type == "torch":
                masks, _, low_res_logits, mk_feature = model_B.fwd.predictor.predict(**indata.ort_inputs)
            else:
                masks, score, low_res_logits = ort_session.run(None, indata.ort_inputs)
                masks = masks[0]

            masks = masks > 0
            mask_img = show_mask_cv(masks[0], img_crop, random_color=False)
            ME.img = mask_img
            indata.update(low_res_logits)
        # 框选
        elif k == ord('b'):
            label, rect = ME("rect")
            indata.add_box(rect)
            if fwd_type == "torch":
                masks, _, low_res_logits, mk_feature = model_B.fwd.predictor.predict(**indata.ort_inputs)
            else:
                masks, score, low_res_logits = ort_session.run(None, indata.ort_inputs)
                masks = masks[0]
            masks = masks > 0
            mask_img = show_mask_cv(masks[0], img_crop, random_color=False)
            ME.img = mask_img
            indata.update(low_res_logits)
        # 点选
        elif k == ord('n'):
            label, point = ME("point")
            indata.add_point(point, label)
            if fwd_type == "torch":
                masks, _, low_res_logits, mk_feature = model_B.fwd.predictor.predict(**indata.ort_inputs)
            else:
                masks, score, low_res_logits = ort_session.run(None, indata.ort_inputs)
                masks = masks[0]
            masks = masks > 0
            indata.update(low_res_logits)

            mask_img = show_mask_cv(masks[0], img_crop, random_color=False)
            ME.img = mask_img
        # mask选
        elif k == ord('m'):
            label, Contour = ME("mask")
            indata.add_mask(Contour)
            if fwd_type == "torch":
                masks, _, low_res_logits, mk_feature = model_B.fwd.predictor.predict(**indata.ort_inputs)
            else:
                masks, score, low_res_logits = ort_session.run(None, indata.ort_inputs)
                masks = masks[0]
            masks = masks > 0
            indata.update(low_res_logits)

            mask_img = show_mask_cv(masks[0], img_crop, random_color=False)
            ME.img = mask_img

        # 保存当前mask向量
        elif k == ord('a'):
            print("\n请输入要保存的标签：")
            k = cv2.waitKey(0)
            if k == 27:
                continue
            print("\n保存标签为：", k)
            single_input = M_polygroup(str(k), mask=np.uint8(masks[0])).max_polygon
            single_input = single_input.shift_xy(*xywh[:2]).to_dict()
            if fwd_type == "torch":
                single_input["feature"] = mk_feature
            inputs.append(single_input)

            indata.reset_data()
            print("\n开始下一个标注或停止，\n b : 以box的形式标注，\n n : 以point的形式重新标注，\n 其他 : 结束标注")
            k = cv2.waitKey(0)
            # （1）box的形式
            if k == ord('b'):
                label, rect = ME("rect")
                indata.add_box(rect)
            # （2）point的形式
            elif k == ord('n'):
                label, point = ME("point")
                indata.add_point(point, label)
            # （3）mask的形式
            elif k == ord('m'):
                label, Contour = ME("mask")
                indata.add_mask(Contour)
            else:
                break
            if fwd_type == "torch":
                masks, _, low_res_logits, mk_feature = model_B.fwd.predictor.predict(**indata.ort_inputs)
            else:
                masks, score, low_res_logits = ort_session.run(None, indata.ort_inputs)
                masks = masks[0]
            masks = masks > 0
            mask_img = show_mask_cv(masks[0], img_crop, random_color=False)
            ME.img = mask_img
            indata.update(low_res_logits)
        
    # 全图分割
    res = model_B.select_mask(inputs, img_src, xywh)
    shapes_dct = defaultdict(list)
    for sp in res:
        shapes_dct[sp['label']].append(sp)

    score_th = 0.75
    merge_mask(shapes_dct, score_th, img_crop, xywh)
    while True:
        print("当前阈值: ", score_th) 
        print("\n调整阈值，\n = : 增大 0.01，\n - : 减少 0.01，\n q : 结束")
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord('='):
            score_th += 0.01
            merge_mask(shapes_dct, score_th, img_crop, xywh)
        elif k == ord('-'):
            score_th -= 0.01
            merge_mask(shapes_dct, score_th, img_crop, xywh)

    cv2.destroyWindow("image")
    cv2.destroyAllWindows()

def merge_mask(shapes_dct, score_th, img_crop, xywh):
    x, y = xywh[:2]
    for label, shapes in shapes_dct.items():
        print("label: ", label)
        cv2.namedWindow("mask_all_{}".format(label), 0)
        mask_show = np.zeros(img_crop.shape[:2], 'u1')
        for sp in shapes:
            if sp['confidence'] < score_th:
                continue
            parse_shape(sp).shift_xy(-x,-y).draw_on(mask_show)
        show_mask_cv(mask_show, img_crop, random_color=False, winName="mask_all_{}".format(label))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_src')
    parser.add_argument('img_src')
    parser.add_argument('xywh', type=int, nargs=4)
    args = parser.parse_args()
    demo(args.model_src, args.img_src, args.xywh)