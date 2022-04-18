import cv2
import numpy as np
import yolov5.detect as yolo


def roi_transform(image,ul,ur,ll,lr):
    point_matrix = np.float32([ul, ur, ll, lr])
    width = image.shape[0]
    height = image.shape[1]
    new_ul = [0, 0]
    new_ur = [width, 0]
    new_ll = [0, height]
    new_lr = [width, height]
    converted_points = np.float32([new_ul,new_ur,new_ll,new_lr])
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    img_Output = cv2.warpPerspective(image, perspective_transform, (width, height))
    return img_Output


def pipeline(image):
    cv2.imshow('Original image',image)
    cv2.imwrite('out_img/org_img.jpg', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur, 10, 50)
    cv2.imshow('edge', edge)
    cv2.imwrite('out_img/edge_img.jpg', edge)
    roi_img = roi_transform(edge, [190,140], [450,140], [0,480], [640,480])
    lane_detx = roi_transform(image, [190,140], [450,140], [0,480], [640,480])
    lane_dety = np.copy(lane_detx)
    lane_det = np.copy(lane_detx)
    cv2.imshow('ROI', roi_img)
    cv2.imwrite('out_img/roi_img.jpg', roi_img)
    lines = cv2.HoughLinesP(
        roi_img,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=150,  # Min number of votes for valid line
        minLineLength= 50,  # Min allowed length of line
        maxLineGap= 500  # Max allowed gap between line for joining them
    )
    xmax = 240;
    xmin = 240;
    x_lines_max = []
    x_lines_min = []
    x_lines = []
    # Iterate over points
    for points in lines:
        x1, y1, x2, y2 = points[0]
        if abs(y1 - y2) > 50:
            if x1 > 20 or x1 < 20:
                if x1 > xmax:
                    xmax = x1
                    x_lines_max = ([x1,y1,x2,y2])
                elif x1 < xmin:
                    xmin = x1
                    x_lines_min = ([x1,y1,x2,y2])
    if x_lines_max:
        x_lines.append(x_lines_max)
    if x_lines_min:
        x_lines.append(x_lines_min)
    for i in range(len(x_lines)):
        # Extracted points nested in the list
        x1 = x_lines[i][0]
        y1 = x_lines[i][1]
        x2 = x_lines[i][2]
        y2 = x_lines[i][3]
        # Draw the lines joing the points
        # On the original image
        cv2.line(lane_detx, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("lane detx", lane_detx)
    cv2.imwrite('out_img/lane_detx.jpg', lane_detx)
    ymax = 320
    ymin = 320
    y_lines_max =[]
    y_lines_min =[]
    y_lines = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        if abs(y1 - y2) < 20:
            if y1 > 20:
                if y1 > ymax:
                    ymax = y1
                    y_lines_max = ([x1,y1,x2,y2])
                elif y1 < ymin:
                    ymin = y1
                    y_lines_min = ([x1,y1,x2,y2])
    if y_lines_max:
        y_lines.append(y_lines_max)
    if y_lines_min:
        y_lines.append(y_lines_min)
    for i in range(len(y_lines)):
        # Extracted points nested in the list
        x1 = y_lines[i][0]
        y1 = y_lines[i][1]
        x2 = y_lines[i][2]
        y2 = y_lines[i][3]
        # Draw the lines joing the points
        # On the original image
        cv2.line(lane_dety, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("lane dety", lane_dety)
    cv2.imwrite('out_img/lane_dety.jpg', lane_dety)
    if len(x_lines) == 2:
        overlay = lane_det.copy()
        x_oc = int((x_lines[0][2] + x_lines[1][2]) / 2)
        pts = np.array([[x_lines[0][0], x_lines[0][1]], [x_lines[0][2], x_lines[0][3]], [x_lines[1][0], x_lines[1][1]],
                        [x_lines[1][2], x_lines[1][3]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        overlay = cv2.fillPoly(overlay, pts=[pts], color=(0, 255, 0))
        alpha = 0.4  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        lane_det = cv2.addWeighted(overlay, alpha, lane_det, 1 - alpha, 0)
        cv2.line(lane_det, (320, 640), (x_oc, 0), (255, 255, 0), 2)
        angle = np.degrees(np.arctan((abs(320 - x_oc))/640))
        print("Angle: ", angle)

    if len(y_lines) == 2:
        overlay = lane_det.copy()
        pts = np.array([[y_lines[1][2], y_lines[1][3]], [y_lines[1][0], y_lines[1][1]], [y_lines[0][0], y_lines[0][1]],
                        [y_lines[0][2], y_lines[0][3]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        overlay=cv2.fillPoly(overlay, pts=[pts], color =(255,255,0))
        lane_det = cv2.addWeighted(overlay, alpha, lane_det, 1 - alpha, 0)
    cv2.imshow("lane det", lane_det)
    cv2.imwrite('out_img/lane_det.jpg', lane_det)




"""
# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    pipeline(frame)
    cv2.imwrite('frame.jpg', frame)
    yolo.run(source='frame.jpg')

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
"""
import cv2
img = cv2.imread('sim_lanes_w_car.jpg') # load a dummy image
while(1):
    pipeline(img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
