from ultralytics import YOLO
import time
import math
import tkinter
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def processOneImage(specifiedRegions, boundingBoxes, timeSeries):

    # Assumes boxes given in format [x1,y1,x2,y2]
    regionCounts = []
    for i in range(len(specifiedRegions)):
        regionCounts.append(0)
    for box in boundingBoxes:
        for i in range(len(specifiedRegions)):
            if isBoundingBoxInRegion(box, specifiedRegions[i]):
                regionCounts[i] += abs(box[3].item() - box[1].item()) ** 2
                break

    if not timeSeries:
        for i in range(len(specifiedRegions)):
            timeSeries.append([regionCounts[i]])
    else:
        for i in range(len(specifiedRegions)):
            timeSeries[i].append(regionCounts[i])

    return timeSeries


def isBoundingBoxInRegion(box, region):

    # Assumes boxes given in format [x1,y1,x2,y2] and regions given in format [number,x1,y1,x2,y2]
    center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]  # Average x and y boundaries of bounding box to get its center
    if (center[0].item() <= region[0] and center[0].item() >= region[2] or center[0].item() >= region[0] and center[0].item() <= region[2]) and (
            center[1].item() <= region[1] and center[1].item() >= region[3] or center[1].item() >= region[1] and center[1].item() <= region[3]):
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

    # Arbitrarily Chosen Values
    occupancyValueLimit = 200
    occupancyShortTermIncreaseLimit = 100
    occupancyLongTermIncreaseLimit = 150

    imageLength = 500
    imageWidth = 700

    model = YOLO("yolov8n.pt")  # model
    results = model("C:/Users/CPS/Pictures/capstone image collections/datasets/Yard 4 3-6-23")  # predict on an image
    #time.sleep(5)
    boxes = []
    for res in results:
        boxes = []
        for i in range(len(res.boxes.cls)):
            if res.boxes.cls[i] == 0:
                boxes.append(res.boxes.xyxy[i])

        numHorizontalBoxes = 7
        numVerticalBoxes = 5

        if not specifiedRegions:
            for i in range(numVerticalBoxes):
                for j in range(numHorizontalBoxes):
                    specifiedRegions.append([int(j*imageWidth/numHorizontalBoxes), int(i*imageLength/numVerticalBoxes), int((j+1)*imageWidth/numHorizontalBoxes), int((i+1)*imageLength/numVerticalBoxes)])

        timeSeries = processOneImage(specifiedRegions, boxes, timeSeries)

        generateWarnings(specifiedRegions, timeSeries, occupancyValueLimit, occupancyShortTermIncreaseLimit, occupancyLongTermIncreaseLimit)

    for i in range(len(timeSeries)):
        plt.subplot(numVerticalBoxes, numHorizontalBoxes, i + 1)
        plt.plot(timeSeries[i])
        #plt.title('Region (' + str(specifiedRegions[i][0]) + ', ' + str(specifiedRegions[i][1]) +
        #          '), (' + str(specifiedRegions[i][2]) + ', ' + str(specifiedRegions[i][3]) + ')')
    
    plt.show()


main()
time.sleep(1)
print("It is done!")