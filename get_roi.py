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

# importing the module
import cv2

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font,
					1, (255, 0, 0), 2)
		cv2.imshow('image', img)

	# checking for right mouse clicks
	if event==cv2.EVENT_RBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)

	# reading the image
img = cv2.imread('sim_lanes.jpg', 1)

	# displaying the image
cv2.imshow('image', img)

	# setting mouse handler for the image
	# and calling the click_event() function
cv2.setMouseCallback('image', click_event)

	# wait for a key to be pressed to exit
cv2.waitKey(0)

	# close the window
cv2.destroyAllWindows()
