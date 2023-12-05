import abc
from os import listdir
from os.path import splitext


class data(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dataPath, extension):
        self.dataPath = dataPath
        self.extension = extension

    @abc.abstractmethod
    def numberCases(self):
        pass


class VideoData(data):
    def __init__(self, dataPath, extension):
        VideoData.dataPath = dataPath
        VideoData.extension = extension
        # Read all file names in path with designated extension
        VideoData.fileNames = [
            splitext(f)[0] for f in listdir(self.dataPath) if f.endswith(self.extension)
        ]
        VideoData.ignore = {}

    def number_cases(self):
        return len(self.fileNames)

    def ignore_file(self, names):
        for file_name in names:
            if file_name not in VideoData.ignore:
                VideoData.ignore.append(file_name)

    def remove_file(self, names):
        for file_name in names:
            if file_name in VideoData.fileNames:
                VideoData.fileNames.remove(file_name)

    def add_file(self, names):
        for file_name in names:
            if file_name not in VideoData.fileNames:
                VideoData.fileNames.append(file_name)

    def update_all(self):
        VideoData.fileNames = [
            splitext(f)[0] for f in listdir(self.dataPath) if f.endswith(self.extension)
        ]

class TableData(data):
    def __init__(self, dataPath, extension):
        TableData.dataPath = dataPath
        TableData.extension = extension

    def numberCases(self):
        pass
    
    # takes self, returns data in array form
    def getArray(self):
        opened = open(self.dataPath)
        readed = opened.read()

        # make string table into array table
        lines = readed.split('\n')
        data = [line.split(',') for line in lines]

        # convert string numbers to float
        for i in range(1, len(data)):
            data[i] = [float(x) for x in data[i]]

        return data
    
def test():
    print("TEST FUNCTION RAN")