import random
from random import randint
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from keras.models import load_model
from keras import backend
import seaborn as sns
# ~ Ahmad Alaziz
pd.set_option('display.max_columns', None)
np.set_printoptions(formatter={'float_kind': '{:f}'.format})

# Choose program mode: (X or ANY)
#  X :for training/testing an X class model, that will decide if an X class flare is predicted to hit within 24 hours
# ANY :for training/testing an 'Any' class model, that will decide if a solar flare is predicted to hit within 24 hours
mode = 'X'

# default names based on mode
if mode == 'X':
    def_model_name = "modelX.h5"
    def_dataset = "X_dataset.csv"

elif mode == 'ANY':
    def_model_name = "modelAny.h5"
    def_dataset = "dataset.csv"


def export_df(df):
    # Export df to csv for data analysis
    compression_opts = dict(method='zip', archive_name='out.csv')
    df.to_csv('out.zip', index=False, compression=compression_opts)


# normalize all our data
def normalize_df(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns)

# normalize one column
def normalize_one_column(df, column_name):
    result = df.copy()
    max_value = df[column_name].max()
    min_value = df[column_name].min()
    result[column_name] = (df[column_name] - min_value) / (max_value - min_value)
    return result[column_name]


def data_analysis(df):
    # Checking for Multicollinearity
    corr = df.corr()
    sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)

    # Let's get a quick summary of our dataset
    summary = df.describe()
    # add the standard deviation metric
    summary.loc['+3_std'] = summary.loc['mean'] + (summary.loc['std'] * 3)
    summary.loc['-3_std'] = summary.loc['mean'] - (summary.loc['std'] * 3)
    print(summary)


def create_random_model(X, Y):
    # to avoid cluttering
    backend.clear_session()
    # Generate random numbers inorder to create and train a random model
    random_numbers = [randint(400, 2550), randint(3, 29), randint(3, 25), randint(3, 23), random.uniform(0.1, 0.3)]
    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=random_numbers[4],
                                                        random_state=random_numbers[1])

    # create our neural network
    model = Sequential()
    model.add(Dense(random_numbers[1], input_dim=10, activation='relu'))
    model.add(Dense(random_numbers[2], activation='relu'))
    model.add(Dense(random_numbers[2], activation='relu'))
    model.add(Dense(random_numbers[3], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # fit
    model.fit(x_train, y_train, epochs=random_numbers[0], batch_size=random_numbers[2],
              validation_data=(x_test, y_test), verbose=0)
    return model, x_test, y_test


# create a specified number of random models and pick the best one
def train(number_of_models, X, Y, model_name):
    print('creating {} random models,training them and saving the best one'.format(number_of_models))
    highest_accuracy = 0

    num = number_of_models
    while number_of_models >= 0:
        print("Creating and testing model number: {}/{}".format(num - number_of_models, num))
        result = create_random_model(X, Y);
        model = result[0]
        x_test = result[1]
        y_test = result[2]

        accuracy = model.evaluate(x_test, y_test)
        if accuracy[1] > highest_accuracy:
            highest_accuracy = accuracy[1]
            model.save(model_name)

        number_of_models -= 1
    print("highest accuracy: %.2f%% " % (highest_accuracy * 100))

# test the accuracy of our model
def test(saved_model_name):
    model = load_model(saved_model_name)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=531)
    acc = model.evaluate(x_test, y_test)
    print("model accuracy is: %.2f %% " % (acc[1] * 100))

# predict result given input and saved model
def predict(input_values, saved_model_name):
    model = load_model(saved_model_name)
    return np.round(model.predict(np.array([input_values])))

# check if X_class solar flares are all clear
def all_clear_x_class(input_values, saved_x_class_model):
    if predict(input_values, saved_x_class_model) == 0:
        print("All Clear!, No X class solar flares are expected to hit in the next 24 hours! ")
    else:
        print("Watch Out!, an X class is expected to hit in the next 24 hours")

# check if solar flares of any class are all clear
def all_clear_any_class(input_values, saved_any_class_model):
    if predict(input_values, saved_any_class_model) == 0:
        print("All Clear!, No solar flares of any class are expected to hit in the next 24 hours! ")
    else:
        print("Watch Out!, a solar flare is expected to hit in the next 24 hours")


# header names
headers = ["class", "largestSpot", "spotDistribution", "activity", "evolution", "previousActivity", "complex",
           "complexOnPath", "area", "largestSpotArea", "c-class", "m-class", "x-class"]

# importing dataset
df = pd.read_csv(def_dataset, header=None, names=headers)

# transforming strings(type object) to floating point numbers
ord_enc = OrdinalEncoder()
df["class"] = ord_enc.fit_transform(df[["class"]])
df["largestSpot"] = ord_enc.fit_transform(df[["largestSpot"]])
df["spotDistribution"] = ord_enc.fit_transform(df[["spotDistribution"]])

if mode == "X":
    # exporting dataframe for testing purposes
    export_df(df)

    # splitting into input and output
    X = df.iloc[:, 0:10]
    Y = df.iloc[:, 12:13]

    # train the model
    # train(10, X, Y,def_model_name)

    # test the model
    test(def_model_name)

    # Checking if all is clear for the following input
    input_vals = [5, 0, 3, 1, 3, 1, 1, 1, 1, 1]
    all_clear_x_class(input_vals, def_model_name)
elif mode == "ANY":
    # sum up occurrences of all classes
    df['sum-class'] = df['c-class'] + df['m-class'] + df['x-class']
    df['sum-class'] = df['sum-class'].astype(bool).astype(int)
    df['sum-class'] = normalize_one_column(df, 'sum-class')
    # df = normalizeAll(df)

    # exporting dataframe for testing purposes
    export_df(df)

    # splitting into input and output
    X = df.iloc[:, 0:10]
    Y = df.iloc[:, 13:14]

    # train the model
    # train(10, X, Y,def_model_name)

    # test the model
    test(def_model_name)

    # Checking if all is clear for the following input
    input_vals = [5, 0, 3, 1, 3, 1, 1, 1, 1, 1]
    all_clear_any_class(input_vals, def_model_name)
