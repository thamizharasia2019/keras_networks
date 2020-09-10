import cv2
f=cv2.imread('1.bmp')
cv2.imshow('org',f)

f1=cv2.imread('1a.bmp')
cv2.imshow('mask',f1)

grayImage = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
cv2.imshow('mask1',grayImage)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 64, 255, cv2.THRESH_BINARY)
cv2.imshow('Black white image', blackAndWhiteImage)


print(blackAndWhiteImage.shape)
h =int(blackAndWhiteImage.shape[0])

w= int(blackAndWhiteImage.shape[1])

#print(blackAndWhiteImage[:20])
for i in range(h):
  for j in range(w):
    if blackAndWhiteImage[i][j]==255:
       f[i][j][0]=0
       f[i][j][1]=0
       f[i][j][2]=0
cv2.imshow('new',f)
cv2.waitKey(0)
cv2.destroyAllWindows()

