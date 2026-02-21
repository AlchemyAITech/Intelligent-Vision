import os
from copy import deepcopy

import numpy as np
import onnxruntime

from MyUtils import ltrb2xywh
from forward import Forward_SAM
from mpAI.common.utils.image_opt import MpImage
import cv2
from utils import MouseEvent, show_mask_cv


class inputData():
    def __init__(self, image_embedding, orig_im_size):
        self.image_embedding = image_embedding
        self.orig_im_size = np.array(orig_im_size, dtype=np.float32)
        self.reset_data()

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = self.orig_im_size
        new_h, new_w = self.get_preprocess_shape(
            self.orig_im_size[0], self.orig_im_size[1], 1024
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def reset_data(self):
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        onnx_coord = np.array([[0.0, 0.0]])[None, :, :].astype(np.float32)
        onnx_label = np.array([-1])[None, :].astype(np.float32)
        self.ort_inputs = {
            "image_embeddings": self.image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": self.orig_im_size
        }

    def add_point(self, point, label):
        input_point = self.apply_coords(np.array([point]))[None, :, :].astype(np.float32)
        input_label = np.array([label])[None, :].astype(np.float32)
        self.ort_inputs["point_coords"] = np.concatenate([input_point, self.ort_inputs["point_coords"]], axis=1)
        self.ort_inputs["point_labels"] = np.concatenate([input_label, self.ort_inputs["point_labels"]], axis=1)

    def add_box(self, box):
        input_box = self.apply_coords(np.array(box).reshape(2, 2))[None, :, :].astype(np.float32)
        input_label = np.array([2,3])[None, :].astype(np.float32)
        if self.ort_inputs["point_labels"][0, -1] == -1:
            self.ort_inputs["point_coords"] = np.concatenate([self.ort_inputs["point_coords"][:,:-1,:], input_box], axis=1)
            self.ort_inputs["point_labels"] = np.concatenate([self.ort_inputs["point_labels"][:,:-1], input_label], axis=1)
        else:
            self.ort_inputs["point_coords"] = np.concatenate([self.ort_inputs["point_coords"][:,:-2,:], input_box], axis=1)
            self.ort_inputs["point_labels"] = np.concatenate([self.ort_inputs["point_labels"][:,:-2], input_label], axis=1)

    # update mask
    def update(self, mask):
        self.ort_inputs["mask_input"] = mask
        self.ort_inputs["has_mask_input"] = np.ones(1, dtype=np.float32)

    def apply_mask(self):
        new_h, new_w = self.get_preprocess_shape(
            self.orig_im_size[0], self.orig_im_size[1], 1024
        )
        mask = self.ort_inputs["mask_input"][0,0,:new_h//4,:new_w//4]
        old_h, old_w = self.orig_im_size
        mask = cv2.resize(mask, (int(old_w), int(old_h)))
        return (mask>0).astype(np.uint8)




def demo():
    # img_path
    curdir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(curdir, '../testdata/CO_SAM/1.png')
    model_path = os.path.join(curdir, '../testdata/CO_SAM/segmodel')
    emb_path = os.path.join(curdir, '../testdata/CO_SAM/emb_{}')

    # 1：加载后端模型
    model_B = Forward_SAM({'model.pth': os.path.join(curdir, '../models/CO_SAM/version/20230425/model.pth')},
                        '20230425', 'cuda:0')

    # 2：加载图片
    img = MpImage(img_path)

    # 3：前端切块
    img_f = img.read()
    cv2.namedWindow("image", 0)
    cv2.imshow('image', img_f)
    ME = MouseEvent(img_f, title='image')

    label, rect = ME("rect")

    img_crop = img.read_rect(rect)#.astype(np.uint8)
    cv2.imshow('image', img_crop)
    cv2.waitKey(10)

    # 4：后端跑encoder模型，提供npy文件
    rect_ = ltrb2xywh(*[int(np.round(x)) for x in rect])
    print(rect_)
    emb_npy, max_, min_ = model_B.get_npy(img_path, rect_)

    # # save load npy
    # np.save(emb_path.format("data"), emb_npy)
    # np.save(emb_path.format("max"), max_)
    # np.save(emb_path.format("min"), min_)
    #
    # emb_npy = np.load(emb_path.format("data.npy"))
    # max_ = np.load(emb_path.format("max.npy"))
    # min_ = np.load(emb_path.format("min.npy"))

    # emb_npy = emb_npy
    emb_npy = emb_npy.astype(np.float32)/255.*(max_- min_ + 0.00001) + min_
    # emb_npy = emb_npy.astype(np.float32) / 127. - 1
    print("init model done.")

    # 5：前端根据npy文件给出初始mask
    # 初始化onnx模型, 初始化数据有以下两种形式：
    # （1）box的形式
    # （2）point的形式
    ort_session = onnxruntime.InferenceSession(model_path)
    indata = inputData(emb_npy, img_crop.shape[:2])
    ME = MouseEvent(img_crop, title='image')
    while True:
        k = cv2.waitKey(1)
        # （1）box的形式
        if k == ord('n'):
            label, rect = ME("rect")
            indata.add_box(rect)
            break
        # （2）point的形式
        elif k == ord('m'):
            label, point = ME("point")
            indata.add_point(point, label)
            break
    masks, _, low_res_logits = ort_session.run(None, indata.ort_inputs)
    masks = masks > 0
    print(masks.shape)
    mask_img = show_mask_cv(masks[0,0], img_crop, random_color=False)
    ME.img = mask_img
    indata.update(low_res_logits)

    # 6：结合mask的修正
    #（1）points输入：0：负类，去除区域；1：正类，增加区域
    #（2）box输入：转化为points，label为 2 3，起始点和终止点坐标
    #（3）重置：重置后回到初始状态 5
    #（4）退出：结束标图
    #（5）保存：保存该区域后，同时回到初始状态 5
    while True:
        cv2.imshow('image', ME.img)
        k = cv2.waitKey(100)
        if k == ord('q'):
            break
        elif k == ord('i'):
            indata.reset_data()
            while True:
                k = cv2.waitKey(1)
                # （1）box的形式
                if k == ord('n'):
                    label, rect = ME("rect")
                    indata.add_box(rect)
                    break
                # （2）point的形式
                elif k == ord('m'):
                    label, point = ME("point")
                    indata.add_point(point, label)
                    break
            masks, _, low_res_logits = ort_session.run(None, indata.ort_inputs)
            masks = masks > 0
            mask_img = show_mask_cv(masks[0,0], img_crop, random_color=False)
            ME.img = mask_img
            indata.update(low_res_logits)
        elif k == ord('b'):
            label, rect = ME("rect")
            indata.add_box(rect)
            masks, _, low_res_logits = ort_session.run(None, indata.ort_inputs)
            masks = masks > 0
            mask_img = show_mask_cv(masks[0,0], img_crop, random_color=False)
            ME.img = mask_img
            indata.update(low_res_logits)
        else:
            label, point = ME("point")
            indata.add_point(point, label)
            masks, _, low_res_logits = ort_session.run(None, indata.ort_inputs)

            indata.update(low_res_logits)

            mask_img = show_mask_cv(indata.apply_mask(), img_crop, random_color=False)
            ME.img = mask_img

            # masks = masks > 0
            # mask_img = show_mask_cv(masks[0,0], img_crop, random_color=False)
            # ME.img = mask_img
    cv2.destroyWindow("image")
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    demo()