#!/usr/bin/python
import numpy as np
import cv2
import struct
import os


class DigitEntry:
    def __init__(self, lab, offset, data):
        self.label = lab
        self.offset = offset
        self.data = data

    def getName(self):
        return "d{:1d}-0x{:0>8x}".format(self.label, self.offset)


class DigitStorage:
    IMG_COLS = 28
    IMG_ROWS = 28

    SHEET_WIDTH = 644
    SHEET_HEIGHT = 476

    def __init__(self):
        self.entry = []

        rCnt = self.SHEET_WIDTH/self.IMG_COLS
        cCnt = self.SHEET_HEIGHT/self.IMG_ROWS

        count = rCnt*cCnt

        self.IMGS_PER_SHEET = count

    def store(self, entry):
        """
        Add entry (DigitEntry) to this storage.
        """
        self.entry.append(entry)

    def getSheet(self, n=0):
        """
        Get a sheet of 640x480 size full of digit entries.
        """

        sheet = np.zeros((self.SHEET_HEIGHT, self.SHEET_WIDTH), dtype=np.uint8)

        rows = self.SHEET_HEIGHT/self.IMG_ROWS
        cols = self.SHEET_WIDTH/self.IMG_COLS

        cursor = n*self.IMGS_PER_SHEET
        count = len(self.entry)

        for row in range(rows):
            for col in range(cols):
                if cursor < count:
                    y_start = row*self.IMG_ROWS
                    y_end = y_start+self.IMG_ROWS

                    x_start = col*self.IMG_COLS
                    x_end = x_start+self.IMG_COLS

                    data = self.entry[cursor].data
                    sheet[y_start:y_end, x_start:x_end] = data
                    cursor += 1

        return sheet

    def sheetCount(self):
        """
        Return how many full sheets do you have
        """

        return len(self.entry)/self.IMGS_PER_SHEET


class Compiler:
    LABEL_MAGIC = 2049
    IMAGE_MAGIC = 2051

    def __init__(self, images, labels):
        """
        Read images and labels files,
        create digit storage for each digit type,
        start reading data and populate storages.
        """

        self.storage = [DigitStorage() for i in range(10)]

        self.fImages = open(images, "rb")
        self.fLabels = open(labels, "rb")

    def read(self, limit=16):
        self.readHeaders()

        cnt = 0
        while any([len(s.entry) < limit for s in self.storage]):
            self._readSingle(cnt)
            cnt += 1

    def readSheets(self, limit=1):
        self.readHeaders()

        cnt = 0
        while any([s.sheetCount() < limit for s in self.storage]):
            self._readSingle(cnt)
            cnt += 1

    def _readInt(self, f):
        """
        An integer is stored in 4 bytes.
        Read them, combine and return as integer.
        """
        return struct.unpack(">i", f.read(4))[0]

    def readHeaders(self):
        """
        Read data from headers of both label and images files.
        """
        self.labelMagic = self._readInt(self.fLabels)
        self.labelCount = self._readInt(self.fLabels)

        self.imageMagic = self._readInt(self.fImages)
        self.imageCount = self._readInt(self.fImages)

        self.dRows = self._readInt(self.fImages)
        self.dCols = self._readInt(self.fImages)

        self.headerLen = self.fImages.tell()

        retval = 0

        if(self.labelMagic != self.LABEL_MAGIC):
            retval = 1

        if(self.imageMagic != self.IMAGE_MAGIC):
            retval = 1

        if(self.labelCount != self.imageCount):
            retval = 1

        return retval

    def _getOffset(self, cnt):
        return self.headerLen + cnt*self.dRows*self.dCols

    def _readSingle(self, cnt):
        lab = self.readLabel()
        img = self.readImage()

        offset = self._getOffset(cnt)
        entry = DigitEntry(lab, offset, img)

        self.storage[lab].store(entry)
        cnt += 1

    def readLabel(self):
        return ord(self.fLabels.read(1))

    def readImage(self):
        data = self.fImages.read(self.dRows * self.dCols)
        data = [255 - ord(d) for d in data]
        arr = np.array(data, dtype=np.uint8).reshape((self.dRows, self.dCols))
        return arr

    def storeSheets(self, n=1):
        """
        For each storage,
        Get n sheets of digits
        And save each sheet as an image.
        """

        for d in range(10):
            for sheet in range(n):
                filename_fmt = "digit_sheets/digit_{}/digit_{}-sheet_{}.bmp"
                filename = filename_fmt.format(d, d, sheet)

                dirname = os.path.dirname(filename)

                if os.path.exists(filename):
                    os.remove(filename)

                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                data = self.storage[d].getSheet(sheet)
                cv2.imwrite(filename, data)

    def storeSeparated(self, n=16):
        """
        Create a folder for each storage,
        And extract n entries from each storage,
        And store them in appropriate folders
        each entry in it's own file.
        """

        for d in range(10):
            for digit in self.storage[d].entry[:n]:
                filename_fmt = "digit_separated/digit_{}/{}.bmp"
                filename = filename_fmt.format(d, digit.getName())

                dirname = os.path.dirname(filename)

                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                if os.path.exists(filename):
                    os.remove(filename)

                cv2.imwrite(filename, digit.data)


def wait():
    ESC_KEY = 27
    while cv2.waitKey(0) != ESC_KEY:
        pass

IMAGES = "train-images.idx3-ubyte"
LABELS = "train-labels.idx1-ubyte"

if __name__ == "__main__":
    c = Compiler(IMAGES, LABELS)

    #c.read(5000)
    #c.storeSeparated(5000)
    c.readSheets(3)
    c.storeSheets(3)
