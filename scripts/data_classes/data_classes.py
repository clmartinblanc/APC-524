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


class video_data(data):
    def __init__(self, dataPath, extension):
        videoData.dataPath = dataPath
        videoData.extension = extension
        # Read all file names in path with designated extension
        videoData.fileNames = [
            splitext(f)[0] for f in listdir(self.dataPath) if f.endswith(self.extension)
        ]
        videoData.ignore = {}

    def number_cases(self):
        return len(self.fileNames)

    def ignore_file(self, names):
        for file_name in names:
            if file_name not in videoData.ignore:
                videoData.ignore.append(file_name)

    def remove_file(self, names):
        for file_name in names:
            if file_name in videoData.fileNames:
                videoData.fileNames.remove(file_name)

    def add_file(self, names):
        for file_name in names:
            if file_name not in videoData.fileNames:
                videoData.fileNames.append(file_name)

    def update_all(self):
        videoData.fileNames = [
            splitext(f)[0] for f in listdir(self.dataPath) if f.endswith(self.extension)
        ]
