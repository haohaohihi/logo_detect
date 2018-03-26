# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import time
import numpy as np

# 参考了https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
# 使用了传统的SIFT进行特征值提取,FLANN进行匹配

MIN_MATCH_COUNT = 10 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

class SiftDetector:
    def __init__(self, templates_path, result_path):
        self.templates_path = templates_path
        self.result_path = result_path
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.database =  self.create_database()
        self.__create_result_folder()
    
    def create_database(self):
        """
        读取模板目录里的所有图片，并计算sift特征值，存储在内存中供后续匹配
        """
        print("初始化，创建匹配数据库中...")
        database = dict()
        for t_path in os.listdir(self.templates_path):
            t_full_path = os.path.join(self.templates_path, t_path)
            img = cv2.imread(t_full_path, 1)
            try:
                kp, des = self.sift.detectAndCompute(img, None)
            except Exception as e:
                print(e)
            classname = t_path.split('.')[0]
            database[classname] = {
                                    'd_kp': kp,
                                    'd_des': des,
                                    'd_img': img,
            }
        print("匹配数据库创建成功")
        return database
    
    def match_one_image(self, test_img_detail):
        """
        将读取并计算了sift特征值的图片，对模板库的图片进行特征匹配
        """
        start_time = time.time()
        best_matches = []
        best_classname = ""
        temp = []
        maybe = []
        for k, v in self.database.items():
            try:
                matches = flann.knnMatch(v.get('d_des'), test_img_detail.get('t_des'), k=2)
            except:
                raise CannotMatchException
            # get good matches
            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            temp.append(len(good))
            maybe.append(k)

            if len(good) > MIN_MATCH_COUNT and len(good) >= len(best_matches):
                best_matches = good
                best_classname = k

        if not best_matches:
            print("测试文件为: %s" % test_img_detail.get('t_image_name'))
            print("未匹配成功，最终匹配特征点数 - %d/%d" % (max(temp), MIN_MATCH_COUNT))
            print("结果可能为 %s" % maybe[temp.index(max(temp))])
            print("花费时间：%s" % (time.time() - start_time))
            print("----------------------------------------------------")
            return False, maybe[temp.index(max(temp))], max(temp) 
        else:
            print("测试文件为: %s" % test_img_detail.get('t_image_name'))
            print("最终匹配特征点数为: %d" % len(best_matches))
            print("匹配成功，结果为: %s" % best_classname)
            print("花费时间：%s" % (time.time() - start_time))
            print("----------------------------------------------------")
            return True, best_classname, len(best_matches), best_matches

    def draw_or_save_match_result(self, test_img_detail, best_classname, best_matches):
        """
        将满足阈值的匹配结果进行对比绘制
        """
        if not best_matches:
            return None

        src_pts = np.float32([ self.database.get(best_classname).get('d_kp')[m.queryIdx].pt for m in best_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ test_img_detail.get('t_kp')[m.trainIdx].pt for m in best_matches ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        try:
            matchesMask = mask.ravel().tolist()
        except:
            print(1)

        h, w, c= self.database.get(best_classname).get('d_img').shape
        pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(test_img_detail.get('t_img'), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(self.database.get(best_classname).get('d_img'), self.database.get(best_classname).get('d_kp'), 
                                img2, test_img_detail.get('t_kp'), best_matches, None, **draw_params)

        
        cv2.imwrite(os.path.join(self.result_path, str(time.time()) + "--" + best_classname + '.jpg'), img3)
    
    def __create_result_folder(self):
        if self.result_path and  not os.path.exists(self.result_path):
            os.mkdir(self.result_path)


def readImg(img_path):
    """
    从路径中读取一个图片文件并计算对应的特征值
    """
    img = cv2.imread(img_path, 1)
    sift = cv2.xfeatures2d.SIFT_create()
    t_kp, t_des = sift.detectAndCompute(img,None)
    img_detail = {
                    't_image_name': img_path,
                    't_img': img,
                    't_kp': t_kp,
                    't_des': t_des,
    }
    return img_detail

def main():
    img_path = sys.argv[1]
    siftDetector = SiftDetector("templates", "result")
    img = readImg(img_path)
    result = siftDetector.match_one_image(img)
    if result[0] is True:
        siftDetector.draw_or_save_match_result(img, result[1], result[-1])

if __name__ == "__main__":
    
    main()