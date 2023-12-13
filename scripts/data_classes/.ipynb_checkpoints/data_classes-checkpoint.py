import abc
from os import listdir
from os.path import splitext
import numpy as np


class data(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data_path, extension):
        self.data_path = data_path
        self.extension = extension

    @abc.abstractmethod
    def number_cases(self):
        pass


class VideoData(data):
    def __init__(self, data_path, save_path, extension):
        VideoData.data_path = data_path
        VideoData.save_path = save_path

        VideoData.extension = extension
        # Read all subfolders in path
        VideoData.data_folders = listdir(self.data_path)
        VideoData.ignore = {}

    def file_names(self, n):
        return [
            splitext(f)[0]
            for f in listdir(self.data_path + "/" + self.data_folders[n])
            if f.endswith(self.extension)
        ]


    def output_files(self):
        return listdir(self.save_path)

    def number_cases(self):
        return len(self.data_folders)

    def ignore_file(self, names):
        for file_name in names:
            if file_name not in VideoData.ignore:
                VideoData.ignore.append(file_name)

    def remove_file(self, names):
        for file_name in names:
            if file_name in VideoData.file_names:
                VideoData.file_names.remove(file_name)

    def add_file(self, names):
        for file_name in names:
            if file_name not in VideoData.file_names:
                VideoData.file_names.append(file_name)

    def update_all(self):
        VideoData.file_names = [
            splitext(f)[0]
            for f in listdir(self.data_path)
            if f.endswith(self.extension)
        ]


    def run_script_n(self, script):
        for n in np.arange(0, self.number_cases(), 1):
            script(self, n)

    def run_script(self, script):
        script(self)



class TableData(data):
    def __init__(self, data_path, extension):
        TableData.data_path = data_path
        TableData.extension = extension

    def number_cases(self):
        pass

    # takes self, returns data in array form
    def get_array(self):
        opened = open(self.data_path)
        readed = opened.read()

        # make string table into array table
        lines = readed.split("\n")
        data = [line.split(",") for line in lines]

        # convert string numbers to float
        for i in range(1, len(data)):
            data[i] = [float(x) for x in data[i]]

        return data


def test():
    print("TEST FUNCTION RAN")
