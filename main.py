from ultralytics import YOLO
import time
import math
import cv2 as cv
import numpy as np

def processOneImage(specifiedRegions, boundingBoxes, timeSeries):

    # Assumes boxes given in format [number,x1,y1,x2,y2]
    regionCounts = np.zeros(len(specifiedRegions))
    for box in boundingBoxes:
        for i in range(len(specifiedRegions)):
            if isBoundingBoxInRegion(box, specifiedRegions[i]):
                regionCounts[i] += abs(box[2] - box[4])

    for i in range(len(timeSeries)):
        np.append(timeSeries[i], regionCounts[i])

    return timeSeries


def isBoundingBoxInRegion(box, region):

    # Assumes boxes given in format [x1,y1,x2,y2] and regions given in format [number,x1,y1,x2,y2]
    center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]  # Average x and y boundaries of bounding box to get its center
    if (center[0] <= region[0] and center[0] >= region[2] or center[0] >= region[0] and center[0] <= region[2]) and (
            center[1] <= region[1] and center[1] >= region[3] or center[1] <= region[1] and center[1] >= region[3]):
        return True
    return False


def generateWarnings(specifiedRegions, timeSeries, occupancyValueLimit, occupancyShortTermIncreaseLimit, occupancyLongTermIncreaseLimit):

    for i in range(len(timeSeries)):
        currentValue = timeSeries[i][len(timeSeries[i]) - 1]
        if currentValue >= occupancyValueLimit:
            print("Warning for Region " + str(i) + " (Coordinates: " + str(specifiedRegions[i]) + ") - Current Occupancy Has Exceeded Limit")
        if len(timeSeries[i]) >= 2:
            previousValue = timeSeries[i][len(timeSeries[i]) - 2]
            if currentValue - previousValue >= occupancyShortTermIncreaseLimit:
                print("Warning for Region " + str(i) + " (Coordinates: " + str(
                    specifiedRegions[i]) + ") - Short-Term Occupancy Increase Has Exceeded Limit")
            if len(timeSeries[i]) >= 6:
                lookBackValue = timeSeries[i][len(timeSeries[i]) - 6]
                if currentValue - lookBackValue >= occupancyLongTermIncreaseLimit:
                    print("Warning for Region " + str(i) + " (Coordinates: " + str(
                        specifiedRegions[i]) + ") - Long-Term Occupancy Increase Has Exceeded Limit")


def main():

    specifiedRegions = []  # List containing lists - each element is a list corresponding to one region and contains its coordinates
    timeSeries = []
    for i in range(len(specifiedRegions)):
        timeSeries.append(np.empty(shape=(0, 0)))

    # Arbitrarily Chosen Values
    occupancyValueLimit = 100
    occupancyShortTermIncreaseLimit = 10
    occupancyLongTermIncreaseLimit = 50

    model = YOLO("yolov8n.pt")  # model
    results = model("C:/Users/CPS/Pictures/capstone image collections/datasets/Yard 3 3-6-2023/image30.png")  # predict on an image
    boxes = []
    for res in results:
        for i in range(len(res.boxes.cls)):
            if res.boxes.cls[i] == 0:
                boxes.append(res.boxes.xyxy[i])

        if not specifiedRegions:
            imageLength = 700
            imageWidth = 700
            specifiedRegions.append([0, 0, int(imageWidth/2), int(imageLength/2)])
            specifiedRegions.append([0, int(imageLength / 2), int(imageWidth / 2), imageLength-1])
            specifiedRegions.append([int(imageWidth / 2), 0, imageWidth-1, int(imageLength/2)])
            specifiedRegions.append([int(imageWidth / 2), int(imageLength / 2), imageWidth-1, imageLength-1])

        timeSeries = processOneImage(specifiedRegions, boxes, timeSeries)

        generateWarnings(specifiedRegions, timeSeries, occupancyValueLimit, occupancyShortTermIncreaseLimit, occupancyLongTermIncreaseLimit)
