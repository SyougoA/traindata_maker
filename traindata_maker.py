# coding: utf-8
import cv2

if __name__ == "__main__":
    names = ["使用するディレクトリ名"]
    cascade = cv2.CascadeClassifier("使用するカスケード.xml")
    for name in names:
        image = cv2.imread("hoge" + name + "/face" + ".jpg")# 読み取るディレクトリ先
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        if len(facerect) > 0:
            # 顔を検出する矩形
            for i in range(len(facerect)):
                rect = facerect[i]
                # 検出した訓練データを保存するディレクトリ先
                cv2.imwrite("train_data/" + name + "/face" + str(i) + ".jpg",
                            image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])

        else:
            print("no face")