from ultralytics import YOLO
import time
import math
import cv2 as cv
import numpy as np


def processOneImage(specifiedRegions, boundingBoxes, timeSeries, computeVectors):
    # Assumes boxes given in format [number,x1,y1,x2,y2]
    regionCounts = np.zeros(len(specifiedRegions))
    for box in boundingBoxes:
        for i in range(len(specifiedRegions)):
            if isBoundingBoxInRegion(box, specifiedRegions[i]):
                regionCounts[i] += abs(box[2] - box[4])

    for i in range(len(timeSeries)):
        np.append(timeSeries[i], regionCounts[i])

    if not computeVectors:
        return timeSeries

    vectors = []  # Placeholder for computation of vectors

    return timeSeries, vectors


def isBoundingBoxInRegion(box, region):
    # Assumes boxes given in format [x1,y1,x2,y2] and regions given in format [x1,y1,x2,y2]
    center = [(box[0] + box[2]) / 2,
              (box[1] + box[3]) / 2]  # Average x and y boundaries of bounding box to get its center
    if (center[0] <= region[0] and center[0] >= region[2] or center[0] >= region[0] and center[0] <= region[2]) and (
            center[1] <= region[1] and center[1] >= region[3] or center[1] <= region[1] and center[1] >= region[3]):
        return True
    return False


def prediction(specifiedRegions, vectors, numIterationsAhead, predictedTimeSeries, horizontalDispersion, verticalDispersion):
    # Assumes vectors given in format [x1,y1,x2,y2,height] and regions given in format [x1,y1,x2,y2]
    regionCounts = np.zeros(len(specifiedRegions))
    for vector in vectors:
        predictedCenter = [(vector[2] - vector[0]) * numIterationsAhead + vector[0],
                           (vector[3] - vector[1]) * numIterationsAhead + vector[1]]
        for i in range(len(specifiedRegions)):
            regionCounts[i] += vector[4] ** 2 * computeFractionOfIntersection(specifiedRegions[i], predictedCenter, horizontalDispersion, verticalDispersion)

    for i in range(len(predictedTimeSeries)):
        np.append(predictedTimeSeries[i], regionCounts[i])

    return predictedTimeSeries


def computeFractionOfIntersection(region, center, horizontalDispersion, verticalDispersion):

    # Assumes regions given in format [x1,y1,x2,y2]
    upperLeft = [center[0] - horizontalDispersion/2, center[1] - verticalDispersion/2]
    lowerRight = [center[0] + horizontalDispersion/2, center[1] + verticalDispersion/2]

    if upperLeft[0] >= region[2] or upperLeft[1] >= region[3] or lowerRight[0] <= region[0] or lowerRight[1] <= region[1]:
        return 0

    upperLeftBound = upperLeft
    if upperLeft[0] <= region[0]:
        upperLeftBound[0] = region[0]
    if upperLeft[1] <= region[1]:
        upperLeftBound[1] = region[1]

    lowerRightBound = lowerRight
    if lowerRight[0] >= region[2]:
        lowerRightBound[0] = region[2]
    if lowerRight[1] >= region[3]:
        lowerRightBound[1] = region[3]

    intersectionRegion = abs((lowerRightBound[0] - upperLeftBound[0]) * (lowerRightBound[1] - upperLeftBound[1]))
    totalRegion = abs((lowerRight[0] - upperLeft[0]) * (lowerRight[1] - upperLeft[1]))

    return intersectionRegion / totalRegion


def generateWarnings(specifiedRegions, timeSeries, currentDensityLimit, predictedTimeSeries, predictedDensityLimit, predictionsAvailable):
    for i in range(len(timeSeries)):
        currentValue = timeSeries[i][len(timeSeries[i]) - 1]
        if currentValue >= currentDensityLimit:
            print("Warning for Region " + str(i) + " (Coordinates: " + str(
                specifiedRegions[i]) + ") - Current Occupancy Has Exceeded Limit")

    if predictionsAvailable:
        for i in range(len(predictedTimeSeries)):
            predictedValue = predictedTimeSeries[i][len(predictedTimeSeries[i]) - 1]
            if predictedValue >= predictedDensityLimit:
                print("Warning for Region " + str(i) + " (Coordinates: " + str(
                    specifiedRegions[i]) + ") - Predicted Occupancy Has Exceeded Limit")


def main():
    specifiedRegions = []  # List containing lists - each element is a list corresponding to one region and contains its coordinates
    timeSeries = []
    predictedTimeSeries = []
    for i in range(len(specifiedRegions)):
        timeSeries.append(np.empty(shape=(0, 0)))
        predictedTimeSeries.append(np.empty(shape=(0, 0)))

    # Arbitrarily Chosen Values
    currentDensityLimit = 100
    predictedDensityLimit = 100
    numIterationsAhead = 5
    horizontalDispersion = 100
    verticalDispersion = 100

    model = YOLO("yolov8n.pt")  # model
    results = model(
        "C:/Users/CPS/Pictures/capstone image collections/datasets/Yard 3 3-6-2023/image30.png")  # predict on an image
    boxes = []

    for i in range(len(results)):
        for j in range(len(results[i].boxes.cls)):
            if results[i].boxes.cls[j] == 0:
                boxes.append(results[i].boxes.xyxy[j])

        if not specifiedRegions:
            imageLength = 700
            imageWidth = 700
            specifiedRegions.append([0, 0, int(imageWidth / 2), int(imageLength / 2)])
            specifiedRegions.append([0, int(imageLength / 2), int(imageWidth / 2), imageLength - 1])
            specifiedRegions.append([int(imageWidth / 2), 0, imageWidth - 1, int(imageLength / 2)])
            specifiedRegions.append([int(imageWidth / 2), int(imageLength / 2), imageWidth - 1, imageLength - 1])
            for i in range(len(specifiedRegions)):
                timeSeries.append(np.empty(shape=(0, 0)))
                predictedTimeSeries.append(np.empty(shape=(0, 0)))

        if i == 0:
            timeSeries = processOneImage(specifiedRegions, boxes, timeSeries, False)
            generateWarnings(specifiedRegions, timeSeries, currentDensityLimit, predictedTimeSeries,
                             predictedDensityLimit, False)
        else:
            timeSeries, vectors = processOneImage(specifiedRegions, boxes, timeSeries, True)
            predictedTimeSeries = prediction(specifiedRegions, vectors, numIterationsAhead, predictedTimeSeries,
                                             horizontalDispersion, verticalDispersion)
            generateWarnings(specifiedRegions, timeSeries, currentDensityLimit, predictedTimeSeries,
                             predictedDensityLimit, True)