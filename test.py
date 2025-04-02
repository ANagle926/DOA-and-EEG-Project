import psutil

from Regression import EEGRegressor

"""dataset = Dataset2(max_cases=60, srate=128)

dump(dataset, "my_object.joblib")
#dataset = load("my_object.joblib")

x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test
c_train, c_test = dataset.c_train, dataset.c_test
print("c_test shape: ", c_test.shape)
print("y_test shape ",y_test.shape)
seglen = dataset.SEGLEN

# Initialize EEGRegressor object
eeg_regressor = EEGRegressor(x_train, y_train, x_test, y_test, c_test, seglen)
eeg_regressor.preprocess_data()
eeg_regressor.model = keras.models.load_model("eeg_regressor.keras")
eeg_regressor.plot_with_predictions()"""

from pympler import summary, muppy

def print_memory_usage():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

print_memory_usage()
print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
