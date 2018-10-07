# ____________ Get File ids for the data ____________

from os import listdir
import pandas as pd

# ____________ Get the data indexes for the TRAINING data ____________
list_ids_x4 = [f for f in listdir("./DIV2K_train_LR_bicubic/X4")]
list_ids_x4 = sorted(list_ids_x4)
dict_list_x4 = {"ids": list_ids_x4}
df = pd.DataFrame(dict_list_x4)
df.to_csv("train_data_index_x4.csv", index = False)

# ____________ Get the data indexes for the labels TRAINING data ____________
list_ids_x2 = [f for f in listdir("./DIV2K_train_LR_bicubic/X2")]
list_ids_x2 = sorted(list_ids_x2)
dict_list_x2 = {"ids": list_ids_x2}
df = pd.DataFrame(dict_list_x2)
df.to_csv("train_data_index_x2.csv", index = False)

# ____________ Get the data indexes for the VALIDATION data ____________
list_ids_x4 = [f for f in listdir("./DIV2K_valid_LR_bicubic/X4")]
list_ids_x4 = sorted(list_ids_x4)
dict_list_x4 = {"ids": list_ids_x4}
df = pd.DataFrame(dict_list_x4)
df.to_csv("valid_data_index_x4.csv", index = False)

# ____________ Get the data indexes for the labels VALIDATION data ____________
list_ids_x2 = [f for f in listdir("./DIV2K_valid_LR_bicubic/X2")]
list_ids_x2 = sorted(list_ids_x2)
dict_list_x2 = {"ids": list_ids_x2}
df = pd.DataFrame(dict_list_x2)
df.to_csv("valid_data_index_x2.csv", index = False)
