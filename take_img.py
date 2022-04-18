import cv2

cap = cv2.VideoCapture(1) # video capture source camera (Here webcam of laptop)

# reading the input using the camera
result, image = cap.read()

# If image will detected without any error,
# show result
if result:

	# showing result, it take frame name and image
	# output
	cv2.imshow("GeeksForGeeks", image)

	# saving image in local storage
	cv2.imwrite("lane_test.png", image)

	# If keyboard interrupt occurs, destroy image
	# window
	cv2.waitKey(0)
	cv2.destroyWindow("GeeksForGeeks")

# If captured image is corrupted, moving to else part
else:
	print("No image detected. Please! try again")
