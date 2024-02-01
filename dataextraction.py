import cv2
import numpy as np
import pandas as pd
# Load the video
cap = cv2.VideoCapture('data/pavan.mp4')
# Initialize the pose estimation algorithm
net = cv2.dnn.readNetFromTensorflow('dnn_models/pose/coco/graph_opt.pb')
data=[]
frame_number=0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    # Detect the joint positions
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    # Extract the joint positions
    H, W = frame.shape[:2]
    points = []
    for i in range(18):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int(point[0] * W / out.shape[3])
        y = int(point[1] * H / out.shape[2])
        points.append((x, y))
    # Estimate the joint angles
    shoulder_angle = np.arctan2(points[2][1] - points[1][1], points[2][0] - points[1][0]) - np.arctan2(
        points[3][1] - points[2][1], points[3][0] - points[2][0])
    elbow_angle = np.arctan2(points[3][1] - points[2][1], points[3][0] - points[2][0]) - np.arctan2(
        points[4][1] - points[3][1], points[4][0] - points[3][0])
    wrist_angle = np.arctan2(points[4][1] - points[3][1], points[4][0] - points[3][0]) - np.arctan2(
        points[6][1] - points[5][1], points[6][0] - points[5][0])
    # Extract the relevant features
    features = [shoulder_angle, elbow_angle, wrist_angle]
    frame_number=frame_number+1
    data.append(features)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27: break
df=pd.DataFrame(data)
df.to_csv("results/30.csv",header=['Shoulder','elbow','Wrist'],index=False)

   # Do something with the features, e.g., store them in a database or write them to a file
    #print(features)

'''import pandas as pd

df = pd.DataFrame(data)
df.to_csv(outfile_path, index=False)
print('save complete')'''
# df = pd.DataFrame({"Shoulder": shoulder_angle, "elbow": elbow_angle, "Wrist": wrist_angle})
# df.to_csv("gait_parameters.csv", index=False)
