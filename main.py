import cv2
import numpy as np

cv2.waitKey(0)

def lesson3():
    photo = np.zeros((450, 450, 3), dtype='uint8')

    #photo[100:150, 20:280] = 119, 201, 105
    cv2.rectangle(photo, (0, 0), (100,100), (119, 201, 105), thickness=2)
    cv2.line(photo, (0, photo.shape[0] // 2), (photo.shape[1], photo.shape[1] // 2), (119, 201, 105), thickness=3)
    cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 100, (119, 201, 105), thickness=2)
    cv2.putText(photo, 'Alex', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), thickness=1)

    cv2.imshow('Photo', photo)


def lesson2():
    #cap = cv2.VideoCapture('videos/Big_Bqquck_Bunny_1080_10s_30MB.mp4')
    cap = cv2.VideoCapture(0)
    cap.set(3, 500) # 3 is Width
    cap.set(4, 300) # 4 is Height

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
        #img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, 35, 35)

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        cv2.imshow('Result', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def lesson1():
    img = cv2.imread('images/Screenshot 2023-04-30 205848.png')
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    cv2.imshow('Result', img)
    print(img.shape)
