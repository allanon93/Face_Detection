import cv2

# scales images over and over
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# store image as variable and then store a gray version of image in another variable
img = cv2.imread("304477244_455789273261937_7338442656328875061_n.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# store facial detection code as variable
faces = face_cascade.detectMultiScale(gray_img, 
scaleFactor = 1.1,
minNeighbors = 5)

# find corners of face and draw rectangle around it
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 3)

resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

cv2.imshow("Face Detected", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()