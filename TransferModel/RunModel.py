from keras import config

from joblib import dump, load
from keras.src.saving import load_model

from TransferModel.Build_Models import MCDropout
from TransferModel.Train_Finetuning import test_finetuning, mc_dropout_predict
from TransferModel.Train_Pretrain import visualize_cnn
from VitalDBDataset import VitalDBDataset
from Train_Pretrain import train_pretrain, prepare_pretraining_data, test_pretrain
from Train_Finetuning import train_finetuning


#dataset = VitalDBDataset(max_cases=20, srate=128)
#dump(dataset, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

dataset=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/dataset_twenty_cases_SEGLENMID.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
seglen = dataset.SEGLEN
print("SEGLEN is ", seglen)
print("shape of x is ", x_train.shape)
print("shape of y is ", y_train.shape)

# Prepare training data
x_pretrain, y_pretrain = prepare_pretraining_data(x_train)
# Prepare testing data
x_pretest, y_pretest = prepare_pretraining_data(x_test)

#pretrain_model= train_pretrain(x_pretrain, y_pretrain, seglen=seglen)
#test_pretrain(pretrain_model, x_pretest, y_pretest)
#visualize_cnn()

fine_tune_model= train_finetuning(x_train, y_train, seglen=seglen)
test_finetuning(fine_tune_model, x_test, y_test)


