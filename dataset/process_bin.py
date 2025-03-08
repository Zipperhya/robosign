# from serial import Serial
import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import os
import pickle
import pandas as pd


# Change the configuration file name


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar


def readAndParseData18xx_from_file():
    global byteBuffer, byteBufferLength

    MMWDEMO_UART_MSG_DETECTED_POINTS = 1

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    detObj = {}

    startIdx = -1

    for index in range(byteBufferLength):
        if checkMagicPattern(byteBuffer[index:index + 8:1]) == 1:
            startIdx = index
            break

    # print(startIdx)
    if byteBufferLength > 16:

        # Check that startIdx is not empty
        if startIdx!=-1:

            # Remove the data before the first start index
            if startIdx > 0 and startIdx < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx] = byteBuffer[startIdx:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx:]),
                                                                       dtype='uint8')
                byteBufferLength = byteBufferLength - startIdx

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # print(totalPacketLen)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        # print(numTLVs)

        # Read the TLV messages
        for tlvIdx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4

            # print(tlv_type)

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # Initialize the arrays
                x = np.zeros(numDetectedObj, dtype=np.float32)
                y = np.zeros(numDetectedObj, dtype=np.float32)
                z = np.zeros(numDetectedObj, dtype=np.float32)
                velocity = np.zeros(numDetectedObj, dtype=np.float32)

                # print(numDetectedObj)

                for objectNum in range(numDetectedObj):
                    # Read the data for each object
                    x[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                    y[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                    z[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                    velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4

                # Store the data in the detObj dictionary
                detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity": velocity}
                # print(detObj)
                dataOK = 1

        # Remove already processed data
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
    # print(byteBufferLength)
    return dataOK, magicOK, detObj


def checkMagicPattern(data):
    """!
       This function check if data arrary contains the magic pattern which is the start of one mmw demo output packet.

        @param data : 1-demension byte array
        @return     : 1 if magic pattern is found
                      0 if magic pattern is not found
    """
    found = 0
    if (data[0] == 2 and data[1] == 1 and data[2] == 4 and data[3] == 3 and data[4] == 6 and data[5] == 5 and data[
        6] == 8 and data[7] == 7):
        found = 1
    return (found)

def update_2(file_path, save_path=None, save_name =None):

    """
    This function reads the data from the .dat file and saves the data in a csv file.
    param:
    file_path: the path of the .dat file to be read
    save_path: the path to save the csv file
    save_name: the name of the csv file
    """

    frameData = {}
    currentIndex = 0
    # file_path =

    fp = open(file_path, 'rb')

    allBinData = fp.read()

    global byteBuffer, byteBufferLength

    # Constants
    maxBufferSize = 2 ** 20


    # Attention: magic OK should be checked, but it is assumed that the magic number is always correct
    magicOK = 1  # Checks if magic number has been read

    byteVec = np.frombuffer(allBinData, dtype='uint8')
    byteCount = len(byteVec)
    print(file_path)
    print(byteCount)
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:

        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    while (magicOK):
        dataOK, magicOK, detObj = readAndParseData18xx_from_file(allBinData, [0])


        if dataOK:

            frameData[currentIndex] = detObj
            currentIndex += 1

    # print(frameData)

    # if save_path is not None and save_name is not None:

    dict2csv(frameData, save_path, save_name)





def dict2csv(pointdic, Savepath=None, SaveName=None):
    df = pd.DataFrame(columns=['frame', 'DetObj#', 'x', 'y', 'z', 'v'])

    for k, s in pointdic.items():
        frame = np.repeat(k, s['numObj'])
        x = s['x']
        y = s['y']
        z = s['z']
        v = s['velocity']
        obj = np.arange(0, s['numObj'])
        new_arr = np.vstack((frame, obj, x, y, z, v)).transpose()
        df1 = pd.DataFrame(new_arr, columns=['frame', 'DetObj#', 'x', 'y', 'z', 'v'])
        df = pd.concat([df, df1])

    df['frame'] = df['frame'].astype(int)
    df['DetObj#'] = df['DetObj#'].astype(int)
    df = df.reset_index(drop=True)
    # print(df)

    if Savepath is not None and SaveName is not None:
        path = os.path.join(Savepath,SaveName)
        df.to_csv(path, index=None)

    return df

def read_all_dat(FilePath, SavePath):
    # process one word in the directory
    # SavePath should share the similar file structure with FilePath

    global byteBuffer, byteBufferLength

    all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(FilePath) for f in filenames]
    for filepath in all_files:
        filepath = filepath.replace('\\','/')

        file = filepath.split('/')[-1]
        exp = file.split('.')[-1]
        file_name = file.split('.')[0]

        if exp != 'dat':
            continue

        # for some .dat file start with 'xwr' or 'test', we should skip them as they are not formal data
        if file_name.startswith('xwr'):
            continue
        if file_name.startswith('test'):
            continue

        save_name = os.path.split(filepath)[1].split('.')[0] + ".csv"

        # process a .dat file
        update_2(filepath, SavePath, save_name)

        if byteBufferLength != 0:
            byteBufferLength = 0




def read_all_dir(DirPath, SavePath=None):

    # process all words in the directory
    # SavePath should share the similar file structure with DirPath
    global byteBuffer, byteBufferLength
    for dirs in os.listdir(DirPath):

        byteBuffer = np.zeros(2 ** 20, dtype='uint8')
        byteBufferLength = 0

        if os.path.exists(os.path.join(SavePath, dirs)):
            continue
        else:
            os.mkdir(os.path.join(SavePath, dirs))
            read_all_dat(os.path.join(DirPath,dirs),os.path.join(SavePath,dirs))
            # break

        # print(dirs)

def read_root_dir(root_path, SavePath=None):

    # When the dictionary is not well-orgnized, we should use this function to process all the data
    # or somebody wants to process the whole dictionary at one time

    global byteBuffer, byteBufferLength
    for dirpath, dirnames, filenames in os.walk(root_path):

        byteBuffer = np.zeros(2 ** 20, dtype='uint8')
        byteBufferLength = 0

        if len(dirnames) == 0:

            relative_path = os.path.relpath(dirpath, root_path)

            if os.path.exists(os.path.join(SavePath, relative_path)):
                continue
            else:
                os.makedirs(os.path.join(SavePath, relative_path))
                read_all_dat(os.path.join(root_path,relative_path),os.path.join(SavePath,relative_path))
                # break

        # print(dirs)


# -------------------------    MAIN   -----------------------------------------

if __name__ == '__main__':

    byteBuffer = np.zeros(2 ** 20, dtype='uint8')
    byteBufferLength = 0






