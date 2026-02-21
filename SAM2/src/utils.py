import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class MouseEvent(object):
    def __init__(self, img, title='image'):
        self.img = img
        self.title = title
        self.init_param()

    def init_param(self):
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.rect = None
        self.tmp_xy = None
        self.label = -1
        self.points = []  # 用于存储多边形的点


    def onMouse_rect(self, event, x, y, flags, param):
        self.tmp_xy = x,y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix = x
            self.iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect = [self.ix, self.iy, x, y]

    def onMouse_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix = x
            self.iy = y
            self.label = 1
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.ix = x
            self.iy = y
            self.label = 0
    
    def onMouse_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))  # 增加当前点到多边形点列表
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.label = 1  # 结束绘制

    def get_rect(self):
        self.init_param()
        cv2.setMouseCallback(self.title, self.onMouse_rect)
        cv2.imshow(self.title, self.img)

        while self.rect is None:
            im_draw = np.copy(self.img)
            if self.ix !=-1 and self.iy!=-1:
                cv2.rectangle(im_draw, (self.ix, self.iy),
                              self.tmp_xy, (255, 0, 0), 2)

            cv2.imshow(self.title, im_draw)
            _ = cv2.waitKey(10)
        return 1, self.rect

    def get_point(self):
        self.init_param()
        cv2.setMouseCallback(self.title, self.onMouse_point)

        while self.label ==-1:
            # cv2.imshow(self.title, self.img)
            _ = cv2.waitKey(10)
        return self.label , [self.ix, self.iy]
    
    def get_mask(self):
        self.init_param()
        cv2.setMouseCallback(self.title, self.onMouse_mask)

        while self.label == -1:
            im_draw = np.copy(self.img)
            if self.points:
                # 画出多边形
                for i in range(len(self.points) - 1):
                    cv2.line(im_draw, self.points[i], self.points[i + 1], (255, 0, 0), 2)
                if len(self.points) > 2:
                    cv2.line(im_draw, self.points[-1], self.points[0], (255, 0, 0), 2)

            cv2.imshow(self.title, im_draw)
            _ = cv2.waitKey(10)
        return 1, self.points

    def __call__(self, even_type):
        if even_type == "rect":
            return self.get_rect()
        if even_type == "mask":
            return self.get_mask()
        else:
            return self.get_point()

def show_mask_cv(mask, img, random_color=False, winName="image"):
    if random_color:
        color = (np.random.random(3) *255).astype(np.uint8)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]

    mask = mask.reshape(h, w, 1).astype(np.uint8)

    mask_image = mask * color.reshape(1, 1, -1).astype(np.uint8)

    mask_image = (cv2.addWeighted(img, 0.6, mask_image, 0.4, 3) * mask + img *(1-mask)).astype(np.uint8)
    cv2.imshow(winName, mask_image)
    cv2.waitKey(10)
    return mask_image

