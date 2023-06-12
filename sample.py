import cv2
from ViTPose.SeafoodVitKit import VitModel

model = VitModel("./vitPose_model.pth")

img = cv2.imread("./test.jpg")
points = model.vitPred(img)
img = model.visualization(img, points)

cv2.imshow("l", img)
cv2.waitKey(0)