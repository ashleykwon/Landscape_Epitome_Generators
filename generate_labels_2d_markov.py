import numpy as np
from matplotlib import pyplot as pt
import util
import random


def sample_pixels(numSamples, windowSize, allLabels, chosenlabel):
    '''
    allLabels: 2-dimensional matrix with nlcd labels in integers
    '''
    sampled = random.sample(chosenlabel, numSamples)

    neighbors = []

    for pair in sampled:
        x = pair[0]
        y = pair[1]
        neighbors += list(allLabels[x-windowSize//2:x+windowSize//2+1, y-windowSize//2:y+windowSize//2+1].flatten())
    
    return neighbors
    

        
def remove_edge_pixels(labelsDict, windowSize, imgSize):
    for elt in list(labelsDict.keys()):
        labelsDict[elt] = list(zip(labelsDict[elt][0], labelsDict[elt][1]))
        labelsDict[elt] = [pair for pair in labelsDict[elt] if pair[0]-windowSize//2 >= 0 and pair[0]+windowSize//2 + 1 < imgSize[0]
        and pair[1]-windowSize//2 >= 0 and pair[1]+windowSize//2 + 1 < imgSize[1]]
    return labelsDict



def countDiagonal(nlcd, labels):
    diag = list(nlcd.diagonal()) + list(np.flip(nlcd, axis = 1).diagonal())
    for elt in diag:
        labels[elt] += 1
    for category in labels:
        labels[category] /= nlcd.shape[0]
    return labels



def countAll(nlcd, labels):
    for elt in list(nlcd.flatten()):
        labels[elt]+= 1
    for category in labels:
        labels[category] /= (nlcd.shape[0]*nlcd.shape[1])
    return labels



def getNeighbors(pxlLoc, patchSize):
    possibleIdx = [[1,1], [1,0], [1,-1], [0,1], [0,-1], [-1,-1], [-1,0], [-1,1]]
    neighborsToCheck = []
    for i in range(len(possibleIdx)): 
        if pxlLoc[0] + possibleIdx[i][0] < patchSize and pxlLoc[0] + possibleIdx[i][0] >= 0 and pxlLoc[1] + possibleIdx[i][1] < patchSize and pxlLoc[1] + possibleIdx[i][1] >= 0:
            neighborsToCheck.append((pxlLoc[0]+possibleIdx[i][0], pxlLoc[1]+possibleIdx[i][1]))
        else:
            continue
    return neighborsToCheck



def generateImg(neighborInfo, nlcd, patchSize):
    initIdx = (np.random.randint(0,patchSize), np.random.randint(0, patchSize))
    newImg = np.zeros((patchSize, patchSize))
    newImg[initIdx[0], initIdx[1]] = random.choice(list(nlcd.flatten()))

    stack = []
    colored = set()

    stack += getNeighbors(initIdx, patchSize)

    while len(stack) != 0:
        currentIdx = stack.pop()
        if currentIdx not in colored:
            newImg[currentIdx[0],currentIdx[1]] = random.choice(neighborInfo[nlcd[currentIdx[0], currentIdx[1]]])
            colored.add(currentIdx)
            stack += getNeighbors(currentIdx, patchSize)
        else: 
            continue
    return newImg

    


def diagonalsNeighbors(nlcd, labelsAsIdx):
    A = np.zeros((len(labels), len(labels)))
    A_pr = np.zeros((len(labels), len(labels)))
    side = nlcd.shape[0]

    diag = list(zip(list(range(side)), list(range(side))))
    rightOfDiag = [[x,y] for [x,y] in list(zip([n+1 for n  in list(range(side))], list(range(side)))) if x < side]
    belowDiag  =  [[x,y] for [x,y] in list(zip(list(range(side)), [n+1 for n  in list(side)])) if y <side]

    for i in range(len(diag)):
        #get the entry at diag[i]-> get the index of this inn labels
        diagEntry =  nlcd[diag[i][0], diag[i][1]]
        #get the entry at rightOfDiag[i]-> get the index of this inn labels
        rightEntry = nlcd[rightOfDiag[i][0], rightOfDiag[i][1]]
        #get the entry at belowDiag[i]-> get the index of this inn labels
        belowEntry = nlcd[belowDiag[i][0], belowDiag[i][1]]

        #get the corresponding index from labels and +1 at 
        A[labelsAsIdx[diagEntry], labelsAsIdx[rightEntry]] += 1
        #get the corresponding index from labels and +1 at A[idx[diag[i]], idx[belowDiag[i]]] 
        A_pr[labelsAsIdx[diagEntry], labelsAsIdx[rightEntry]] += 1
    
    #divide all entries in A by nlcd
    A /= side
    A_pr /= side
    
    return A, A_pr

def chooseLabels(nlcd, labelsList):
    




#0. set a patch size
#1. take sample a pixel in a category and look at its neighbors (n by n square)
#2. count the number of pixels that belong to each category
#3. divide 2 by n*n
#4. choose categories to represent in a patch
#5. 


if __name__ == "__main__":


    img, lc, nlcd = util.load_data()

    #labeltypes = set(nlcd.flatten()) #cardinality = 14 -> {71, 41, 42, 11, 43, 81, 82, 52, 21, 22, 23, 24, 90, 95}
    nlcd_square = nlcd[:6156, :]
    patch_size = 600
    numSamples = 500
    windowSize = 11
    imgSize = [nlcd.shape[0], nlcd.shape[1]]
    #labels = {11:0, 21:0, 22:0, 23:0, 24:0, 41:0, 42:0, 43:0, 52:0, 71:0, 81:0, 82:0, 90:0, 95:0}

    #initialProbs = countDiagonal(nlcd_square, labels)
    #initialProbs2 =  countAll(nlcd, labels)

    idx = {0: 11, 1: 21, 2: 22, 3: 23, 4:24, 5:41, 6:42, 7:43, 8:52, 9:71, 10:81, 11:82, 12:90, 13:95}


    #Read and save the number of pixels in each category 

    labels = {11: np.where(nlcd == 11), 21: np.where(nlcd == 21), 22: np.where(nlcd == 22),
    23: np.where(nlcd == 23), 24: np.where(nlcd == 24), 41: np.where(nlcd == 41), 42: np.where(nlcd == 42), 43: np.where(nlcd == 43), 
    52: np.where(nlcd == 52), 71: np.where(nlcd == 71), 81: np.where(nlcd == 81), 82: np.where(nlcd == 82),
    90: np.where(nlcd == 90), 95: np.where(nlcd == 95)}

    #labels = {21: np.where(nlcd == 21), 22: np.where(nlcd == 22), 23: np.where(nlcd == 23), 24: np.where(nlcd == 24)}

    noEdgeLabels = remove_edge_pixels(labels, windowSize, imgSize)
    
    generator = dict()

    for label in list(noEdgeLabels.keys()):
        generator[label] = sample_pixels(numSamples, windowSize, nlcd, noEdgeLabels[label])
    
    newLabels = generateImg(generator, nlcd, 200)
    labelsRGB = util.vis_nlcd(newLabels, True)
    labelsRGB2 = labelsRGB.swapaxes(1,2).T
    pt.imsave('new_labels_visualized_developed_areas.png', labelsRGB2)
    print('done')

