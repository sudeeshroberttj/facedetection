import cv2

#Create a Cascade Classifier Object

face_cascade = cv2.CascadeClassifier("/Users/mac/PycharmProjects/pythonProject/venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
#Reading the image
img = cv2.imread("family_group.jpg")

#convert to gray
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors=5)

faces = face_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors = 8,
    minSize=(30, 30),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    flags = cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_FIND_BIGGEST_OBJECT|cv2.CASCADE_DO_ROUGH_SEARCH
)

print(type(faces))
print(faces)
print("Found {0} faces!".format(len(faces)))

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    #resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2) ))
    cv2.imshow("Gray",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()