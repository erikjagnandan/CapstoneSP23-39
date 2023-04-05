from ultralytics import YOLO
import time
import math
import tkinter
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort

def processOneImage(specifiedRegions, boundingBoxes, timeSeries, predictedBoundingBoxes, shortTermPredictedTimeSeries):

    # Assumes boxes given in format [x1,y1,x2,y2]
    regionCurrentCounts = []
    for i in range(len(specifiedRegions)):
        regionCurrentCounts.append(0)
    for box in boundingBoxes:
        for i in range(len(specifiedRegions)):
            if isBoundingBoxInRegion(box, specifiedRegions[i]):
                regionCurrentCounts[i] += (box[3].item() - box[1].item()) ** 2
    
    regionPredictedCounts = []
    for i in range(len(specifiedRegions)):
        regionPredictedCounts.append(0)
    for box in predictedBoundingBoxes:
        for i in range(len(specifiedRegions)):
            if isBoundingBoxInRegion(box, specifiedRegions[i]):
                regionPredictedCounts[i] += (box[3].item() - box[1].item()) ** 2

    if not timeSeries:
        for i in range(len(specifiedRegions)):
            timeSeries.append([regionCurrentCounts[i]])
    else:
        for i in range(len(specifiedRegions)):
            timeSeries[i].append(regionCurrentCounts[i])
    
    if not shortTermPredictedTimeSeries:
        for i in range(len(specifiedRegions)):
            shortTermPredictedTimeSeries.append([regionPredictedCounts[i]])
    else:
        for i in range(len(specifiedRegions)):
            shortTermPredictedTimeSeries[i].append(regionPredictedCounts[i])

    return timeSeries, shortTermPredictedTimeSeries


def prediction(specifiedRegions, vectors, numIterationsAhead, longTermPredictedTimeSeries, horizontalDispersion, verticalDispersion):
    # Assumes vectors given in format [x1,y1,x2,y2,width,height] and regions given in format [x1,y1,x2,y2]
    regionCounts = np.zeros(len(specifiedRegions))
    for vector in vectors:
        predictedCenter = [(vector[2] - vector[0]) * numIterationsAhead + vector[0],
                           (vector[3] - vector[1]) * numIterationsAhead + vector[1]]
        box = [predictedCenter[0] - vector[4] / 2 - horizontalDispersion,
                   predictedCenter[1] - vector[5] / 2 - verticalDispersion,
                   predictedCenter[0] + vector[4] / 2 + horizontalDispersion,
                   predictedCenter[1] + vector[5] / 2 + verticalDispersion]
        for i in range(len(specifiedRegions)):
            if isBoundingBoxInRegion(box, specifiedRegions[i]):
                regionCounts[i] += vector[5] ** 2

    if not longTermPredictedTimeSeries:
        for i in range(len(specifiedRegions)):
            longTermPredictedTimeSeries.append([regionCounts[i]])
    else:
        for i in range(len(specifiedRegions)):
            longTermPredictedTimeSeries[i].append(regionCounts[i])

    return longTermPredictedTimeSeries


def isBoundingBoxInRegion(box, region):

    # Assumes boxes given in format [x1,y1,x2,y2] and regions given in format [number,x1,y1,x2,y2]
    center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]  # Average x and y boundaries of bounding box to get its center
    if (center[0].item() <= region[0] and center[0].item() >= region[2] or center[0].item() >= region[0] and center[0].item() <= region[2]) and (
            center[1].item() <= region[1] and center[1].item() >= region[3] or center[1].item() >= region[1] and center[1].item() <= region[3]):
        return True
    return False


def generateWarnings(specifiedRegions, timeSeries, currentDensityLimit, shortTermPredictedTimeSeries, shortTermPredictedDensityLimit, longTermPredictedTimeSeries, longTermPredictedDensityLimit):
    for i in range(len(timeSeries)):
        currentValue = timeSeries[i][len(timeSeries[i]) - 1]
        if currentValue >= currentDensityLimit:
            print("Warning for Region " + str(i) + " (Coordinates: " + str(
                specifiedRegions[i]) + ") - Current Occupancy Has Exceeded Limit")

    for i in range(len(shortTermPredictedTimeSeries)):
        shortTermPredictedValue = shortTermPredictedTimeSeries[i][len(shortTermPredictedTimeSeries[i]) - 1]
        if shortTermPredictedValue >= shortTermPredictedDensityLimit:
            print("Warning for Region " + str(i) + " (Coordinates: " + str(
                specifiedRegions[i]) + ") - Short Term Predicted Occupancy Has Exceeded Limit")
    
    for i in range(len(longTermPredictedTimeSeries)):
        longTermPredictedValue = longTermPredictedTimeSeries[i][len(longTermPredictedTimeSeries[i]) - 1]
        if longTermPredictedValue >= longTermPredictedDensityLimit:
            print("Warning for Region " + str(i) + " (Coordinates: " + str(
                specifiedRegions[i]) + ") - Long Term Predicted Occupancy Has Exceeded Limit")


def main():

    specifiedRegions = []  # List containing lists - each element is a list corresponding to one region and contains its coordinates
    timeSeries = []
    shortTermPredictedTimeSeries = []
    longTermPredictedTimeSeries = []

    # Arbitrarily Chosen Values
    currentDensityLimit = 300
    shortTermPredictedDensityLimit = 1000
    longTermPredictedDensityLimit = 1000
    numIterationsAhead = 3
    horizontalDispersion = 30
    verticalDispersion = 30

    numHorizontalBoxes = 7
    numVerticalBoxes = 5

    imageLength = 500
    imageWidth = 700

    model = YOLO("yolov8n.pt")  # model
    results = model("C:/Users/CPS/Pictures/capstone image collections/datasets/Yard 4 3-6-23")  # predict on an image
    #time.sleep(10)
    boxes = []
    for i in range(len(results)):
        boxes = []
        tracker = DeepSort(max_age=5)
        bbs = []
        predictedBoxes = []
        vectors = []
        for j in range(len(results[i].boxes.cls)):
            if results[i].boxes.cls[j] == 0:
                bbs.append((results[i].boxes.xywh[j],results[i].boxes.conf[j],0))
                boxes.append(results[i].boxes.xyxy[j])
        tracks = tracker.update_tracks(bbs,frame=results[i].orig_img)

        for j in range(len(tracks)):
            newVector = [0, 0, 0, 0, 0, 0]
            ltrb = tracks[j].to_ltrb()
            newVector[0] = (results[i].boxes.xyxy[j][0].item() + results[i].boxes.xyxy[j][2].item()) / 2
            newVector[1] = (results[i].boxes.xyxy[j][1].item() + results[i].boxes.xyxy[j][3].item()) / 2
            newVector[2] = (ltrb[0] + ltrb[2]) / 2
            newVector[3] = (ltrb[1] + ltrb[3]) / 2
            newVector[4] = abs(results[i].boxes.xyxy[j][2].item() - results[i].boxes.xyxy[j][0].item())
            newVector[5] = abs(results[i].boxes.xyxy[j][3].item() - results[i].boxes.xyxy[j][1].item())
            vectors.append(newVector)
            predictedBoxes.append(ltrb)

        if not specifiedRegions:
            for m in range(numVerticalBoxes):
                for n in range(numHorizontalBoxes):
                    specifiedRegions.append([int(n*imageWidth/numHorizontalBoxes), int(m*imageLength/numVerticalBoxes), int((n+1)*imageWidth/numHorizontalBoxes), int((m+1)*imageLength/numVerticalBoxes)])

        timeSeries, shortTermPredictedTimeSeries = processOneImage(specifiedRegions, boxes, timeSeries, predictedBoxes, shortTermPredictedTimeSeries)
        longTermPredictedTimeSeries = prediction(specifiedRegions, vectors, numIterationsAhead, longTermPredictedTimeSeries, horizontalDispersion,
                                             verticalDispersion)
        generateWarnings(specifiedRegions, timeSeries, currentDensityLimit, shortTermPredictedTimeSeries, shortTermPredictedDensityLimit, longTermPredictedTimeSeries, longTermPredictedDensityLimit)

    for i in range(len(timeSeries)):
        plt.subplot(numVerticalBoxes, numHorizontalBoxes, i + 1)
        plt.plot(timeSeries[i])
        #plt.title('Region (' + str(specifiedRegions[i][0]) + ', ' + str(specifiedRegions[i][1]) +
        #          '), (' + str(specifiedRegions[i][2]) + ', ' + str(specifiedRegions[i][3]) + ')')
    
    plt.show()

    for i in range(len(shortTermPredictedTimeSeries)):
        plt.subplot(numVerticalBoxes, numHorizontalBoxes, i + 1)
        plt.plot(shortTermPredictedTimeSeries[i])
        #plt.title('Region (' + str(specifiedRegions[i][0]) + ', ' + str(specifiedRegions[i][1]) +
        #          '), (' + str(specifiedRegions[i][2]) + ', ' + str(specifiedRegions[i][3]) + ')')
    
    plt.show()

    for i in range(len(longTermPredictedTimeSeries)):
        plt.subplot(numVerticalBoxes, numHorizontalBoxes, i + 1)
        plt.plot(longTermPredictedTimeSeries[i])
        #plt.title('Region (' + str(specifiedRegions[i][0]) + ', ' + str(specifiedRegions[i][1]) +
        #          '), (' + str(specifiedRegions[i][2]) + ', ' + str(specifiedRegions[i][3]) + ')')
    
    plt.show()


main()
time.sleep(1)
print("It is done!")