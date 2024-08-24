import pandas as pd
import tensorflow as tf
print(tf.__version__)


raw_baseball_data = pd.read_csv('./data/dataset_189_baseball.arff', dtype=str) # .arff format is apprently mostly equivalent for .csv for our purposes (look up Weka if really curious)

# expand default pandas display options to make things more clearly visible when printed
pd.set_option('display.max_colwidth', 300)

print(raw_baseball_data.head())


raw_data = pd.read_csv('./data/2002-2018-bc-public-libraries-open-data-csv-v18.2.csv', dtype=str)
print(raw_data.head())
COLUMNS = ["PCT_ELEC_IN_TOT_VOLS","TOT_AV_VOLS"] # lots of columns in this data set, let's just focus on these two
raw_library_data = raw_data[COLUMNS]
print(raw_library_data)