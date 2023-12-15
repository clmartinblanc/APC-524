import data_classes as data_classes


test_instance = data_classes.TableData("scripts/data_classes/test_data.csv", ".csv")
my_array = test_instance.get_array()
print(my_array)
print(type(test_instance))

csv_instance = data_classes.TableData("scripts/video_demo/output/Data_RunA.csv", ".csv")
ar = csv_instance.get_array()
print(ar)
