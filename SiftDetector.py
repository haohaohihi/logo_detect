import os
import cv2
import time
import pickle

# 参考了https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
# 使用了传统的SIFT进行特征值提取,FLANN进行匹配

MIN_MATCH_COUNT = 20 
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
        print("初始化，创建匹配数据库中...")
        database = dict()
        for t_path in os.listdir(self.templates_path):

            t_full_path = os.path.join(self.templates_path, t_path)
            img = cv2.imread(t_full_path, 1)
            try:
                kp, des = self.sift.detectAndCompute(img, None)
            except:
                print(t_path)
            classname = t_path.split('.')[0]
            database[classname] = {
                                    'd_kp': kp,
                                    'd_des': des,
                                    'd_img': img,
            }
        print("匹配数据库创建成功")
        return database
    
    def match_one_image(self, test_img_detail):
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
            return True, best_classname, len(best_matches)

    def match_one_image_1(self, test_img_detail):
        start_time = time.time()
        match_dict = {}

        for k, v in self.database.items():
            match_dict[k] = []
            try:
                matches = flann.knnMatch(v.get('d_des'), test_img_detail.get('t_des'), k=2)
            except:
                raise CannotMatchException
            # get good matches
            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            match_dict[k].append(len(good))

        max_count = 0
        maybe_name = ""
        for k, v in match_dict.items():
            count_temp = sum(v) / len(v)
            if count_temp > max_count:
                max_count = count_temp
                maybe_name = k
                
        if max_count > MIN_MATCH_COUNT:
            best_classname = maybe_name
        else:
            best_classname = ""

        if not best_classname:
            print("测试文件为: %s" % test_img_detail.get('t_image_name'))
            print("未匹配成功，最终匹配特征点数 - %d/%d" % (max_count, MIN_MATCH_COUNT))
            print("结果可能为 %s" % maybe_name)
            print("花费时间：%s" % (time.time() - start_time))
            print("----------------------------------------------------")
            return False, maybe_name, max_count 
        else:
            print("测试文件为: %s" % test_img_detail.get('t_image_name'))
            print("最终匹配特征点数为: %d" % max_count)
            print("匹配成功，结果为: %s" % best_classname)
            print("花费时间：%s" % (time.time() - start_time))
            print("----------------------------------------------------")
            return True, best_classname, max_count


    def draw_or_save_match_result(self, test_img_detail, best_classname, best_matches):
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

    