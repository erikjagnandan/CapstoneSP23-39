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

    # Assumes boxes and regions given in format [number,x1,y1,x2,y2]
    center = [(box[1] + box[3]) / 2, (box[2] + box[4]) / 2]  # Average x and y boundaries of bounding box to get its center
    if (center[0] <= region[1] and center[0] >= region[3] or center[0] >= region[1] and center[0] <= region[3]) and (
            center[1] <= region[2] and center[1] >= region[4] or center[1] <= region[2] and center[1] >= region[4]):
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

    while(True):

        # PLACEHOLDER for Receiving New Image

        # PLACEHOLDER for YOLO computation generating bounding boxes:
        boundingBoxes = []  # List containing lists - each element is a list corresponding to one bounding box and contains its coordinates

        timeSeries = processOneImage(specifiedRegions, boundingBoxes, timeSeries)

        generateWarnings(specifiedRegions, timeSeries, occupancyValueLimit, occupancyShortTermIncreaseLimit, occupancyLongTermIncreaseLimit)
