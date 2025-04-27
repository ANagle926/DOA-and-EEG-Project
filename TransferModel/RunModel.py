from joblib import dump, load
from VitalDBDataset import VitalDBDataset
from Train_Pretrain import train_pretrain
from Train_Finetuning import train_finetuning


#dataset = VitalDBDataset(max_cases=20, srate=128)
#dump(dataset, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases.joblib")

dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen = dataset.SEGLEN


