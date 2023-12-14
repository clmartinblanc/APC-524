from __future__ import annotations
import os
import sys
from data_classes import data_classes

# deleting all files from output folder used later in script to ensure that edge_detection appropiately writes new files every time
previous_test_output_files = os.listdir("tests/output_of_video_test")
for old_data in previous_test_output_files:
    path_to_old_data = os.path.join("tests/output_of_video_test", old_data)
    os.remove(path_to_old_data)


# have edge detection import, was having issues getting python to find it
current_directory = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(current_directory, ".."))
video_scripts_path = os.path.join(root, "scripts", "video_demo", "video_scripts")
sys.path.append(video_scripts_path)
from edge_detection import edge_detection

# creating instance used throughout testing
video_data_instance = data_classes.VideoData(
    "scripts/video_demo/data", "tests/output_of_video_test", ".png"
)


# verifying the instance is appropriately setup
def test_instance():
    # ensuring extension is correct
    assert video_data_instance.extension == ".png"

    # ensuring the data path exists
    assert os.path.exists(video_data_instance.data_path)

    # ensuring the save path exists
    assert os.path.exists(video_data_instance.save_path)


def test_edge_detection():
    # ensuring edge_detection can run on this instance
    # this adds about 10 seconds of testing time, FYI
    video_data_instance.run_script_n(edge_detection)

    # checking that edge_detection wrote output files in correct location with correct name
    assert os.path.exists("tests/output_of_video_test/Data_RunA.csv")
    assert os.path.exists("tests/output_of_video_test/Data_RunB.csv")
    assert os.path.exists("tests/output_of_video_test/Data_RunC.csv")

    # verifying contents of outputs correct
    expected_A = data_classes.TableData(
        "tests/expected_output_of_vid_test/Data_RunA.csv", ".csv"
    )
    expected_A_array = expected_A.get_array()
    expected_B = data_classes.TableData(
        "tests/expected_output_of_vid_test/Data_RunB.csv", ".csv"
    )
    expected_B_array = expected_B.get_array()
    expected_C = data_classes.TableData(
        "tests/expected_output_of_vid_test/Data_RunC.csv", ".csv"
    )
    expected_C_array = expected_C.get_array()

    outputted_A = data_classes.TableData(
        "tests/output_of_video_test/Data_RunA.csv", ".csv"
    )
    outputted_A_array = outputted_A.get_array()
    outputted_B = data_classes.TableData(
        "tests/output_of_video_test/Data_RunB.csv", ".csv"
    )
    outputted_B_array = outputted_B.get_array()
    outputted_C = data_classes.TableData(
        "tests/output_of_video_test/Data_RunC.csv", ".csv"
    )
    outputted_C_array = outputted_C.get_array()

    assert expected_A_array == outputted_A_array
    assert expected_B_array == outputted_B_array
    assert expected_C_array == outputted_C_array
