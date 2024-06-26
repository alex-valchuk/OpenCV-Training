import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot

def main():
    lesson8()

def lesson8():
    img = cv2.imread('images/plate1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
    edges = cv2.Canny(img_filter, 30, 200)

    conturs = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conturs = imutils.grab_contours(conturs)
    conturs = sorted(conturs, key=cv2.contourArea, reverse=True)[:8]

    position = None
    for contur in conturs:
         approx = cv2.approxPolyDP(contur, 10, True)
         if len(approx) == 4:
              position = approx
              break
         
    mask = np.zeros(gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [position], 0, 255, -1)
    bitwise_img = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    crop = gray[x1:x2, y1:y2]

    text = easyocr.Reader(['en'])
    text = text.readtext(crop)

    res = text[0][-2]
    final_image = cv2.putText(img, res, (x1 - 200, y2 + 160), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    final_image = cv2.rectangle(img, (x1, x2), (y1, y2), (0, 255, 0), 2)

    pyplot.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    pyplot.show()

    cv2.waitKey(0)

def lesson71():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = cv2.CascadeClassifier('faces.xml')
        results = faces.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=3)
        
        for (x, y, w, h) in results:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

        cv2.imshow('Result', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def lesson7():
    img = cv2.imread('images/people2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier('faces.xml')
    results = faces.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    
    for (x, y, w, h) in results:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
         
    cv2.imshow("result", img)
    cv2.waitKey(0)

def lesson6():
    photo = cv2.imread('images/Screenshot 2023-04-30 205848.png')
    img = np.zeros(photo.shape[:2], dtype='uint8')

    circle = cv2.circle(img.copy(), (200, 300), 120, 255, cv2.FILLED)
    square = cv2.rectangle(img.copy(), (25, 25), (250, 350), (123, 234, 122), cv2.FILLED)

    img = cv2.bitwise_and(photo, photo, mask=square)
    #img = cv2.bitwise_or(circle, square)
    #img = cv2.bitwise_xor(circle, square)
    #img = cv2.bitwise_not(circle, square)

    cv2.imshow("Result", img)
    cv2.waitKey(0)

def lesson5():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.Canny(img, 30, 30)

        r, g, b = cv2.split(img)
        img = cv2.merge([b, g, r])

        cv2.imshow('Result', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def lesson4():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, 30, 30)

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.flip(img, 1)
        #img = rotate(img, 90)
        #img = transform(img, 30, 200)
        new_img = np.zeros(img.shape, dtype='uint8')

        conturs, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(new_img, conturs, -1, (230, 111, 148), thickness=1)
        cv2.imshow('Result', new_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def rotate(img, angle):
    height, width = img.shape[:2]
    rotationPoint = (height //2, width // 2)

    matrix = cv2.getRotationMatrix2D(rotationPoint, angle, 1)
    return cv2.warpAffine(img, matrix, (height, width))

def transform(img, x, y):
    height, width = img.shape[:2]
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, matrix, (height, width))

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


main()

