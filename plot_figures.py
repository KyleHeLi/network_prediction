#!/usr/bin/env python
# -*- encoding:utf-8 -*-


# Libraries for SARIMA
import glob
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

# Libraries for LSTM
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from numpy import newaxis

import time


def plotData():
    '''Analyze the data from the local directory'''
    files = glob.glob("./6/day*.dat")
    fig = []
    data_org = []
    data_cut = []
    data_len = 0
    n = 125  # 125
    m = 125
    files.sort()
    # print files
    if files:
        # Load bitrate data
        file_num = len(files)

        # Find minimum data length
        data_len = 4 * 48  # getMinFileLen(files)
        # print data_len

        for i in xrange(file_num):
            # print files[i].split("_")[4]
            # data = np.append(data, np.loadtxt(files[i], delimiter=" "))
            data_org = np.append(data_org,
                                 np.loadtxt(files[i], delimiter=" ")[0:data_len])
        for j in xrange(data_len * file_num / 4):
            # data_cut.append(data_org[4 * i] +
            #            data_org[4 * i + 1] +
            #            data_org[4 * i + 2] +
            #            data_org[4 * i + 3])
            data_cut = np.append(data_cut, data_org[4 * j] +
                                 data_org[4 * j + 1] +
                                 data_org[4 * j + 2] +
                                 data_org[4 * j + 3])

        # print data_org
        # print("=============")
        # print data_cut
        # print files[i]

        # Slice the two-dimention array
        # y = data[0:n]
        y = data_cut
        x = np.arange(len(y))
        # fig.append(plt.figure("Packet Number in Day %s" %
        #                       (i + 1), figsize=(10.0, 3.0)))

        # Figure 1
        fig.append(plt.figure("Network Traffic in %s Days" %
                              (i + 1), figsize=(10.0, 3.0)))
        axe = fig[0].add_subplot(111)
        # Draw histogram
        # axe.hist( y, len(y) )
        axe.plot(x, y)
        # axe.bar(x, y)
        axe.set_ylabel("packet number")
        fig[0].tight_layout()

        # predict_arma(number=(i + 1),
        #              index=x[0:m],
        #              data=y[0:m],
        #              original_index=x,
        #              original_data=y)

        # predict_arima(number=(i + 1),
        #               index=x,
        #               data=y)

        # TODO: Comment for now
        rate = 1.0 * 16 / 17
        # predict_sarima(number=(i + 1),
        #                index=x,
        #                data=y,
        #                ratio=rate)

        # errors_sarima = predict_sarima(number=(i + 1),
        #                                index=x,
        #                                data=y,
        #                                ratio=rate)[0]

        # predict at the same time interval
        # errors_rnn_section = predict_rnn_section(number=(i + 1), 
        #                                             index=x, 
        #                                             data=y, 
        #                                             ratio=rate)[0]
        errors_lsarima = predict_improved_lsarima2(number=(i + 1),
                                                  index=x,
                                                  data=y,
                                                  ratio=rate)[0]

        # improved sarima associated with LSTM
        # errors_lsarima = predict_improved_lsarima(number=(i + 1),
        #                                           index=x,
        #                                           data=y,
        #                                           ratio=rate)[0]

        # errors_lstm, a = predict_LSTM_RNN(number=(i + 1),
        #                                   index=x,
        #                                   data=y,
        #                                   ratio=rate)

        # errors_lstm, errors_gru, a = predict_RNN(number=(i + 1),
        #                                          index=x,
        #                                          data=y,
        #                                          ratio=rate)

        print("********************")
        print("Test Score:")
        print("====================")
        print(" \t \t MSE \t\t MAE \t\t MAPE \t\t R2")
        # print("RNN_SECTION:\t %.4f \t %.4f \t %.4f \t %.4f" % (errors_rnn_section[
        #       0], errors_rnn_section[1], errors_rnn_section[2], errors_rnn_section[3]))
        # print("LSTM_RNN:\t %.4f" % (errors_lstm))
        print("LSARIMA:\t %.4f \t %.4f \t %.4f \t %.4f" % (errors_lsarima[
              0], errors_lsarima[1], errors_lsarima[2], errors_lsarima[3]))
        # print("SARIMA:\t %.4f \t %.4f \t %.4f \t %.4f" % (errors_sarima[
        #       0], errors_sarima[1], errors_sarima[2], errors_sarima[3]))
        # print("LSTM_RNN:\t %.4f \t %.4f \t %.4f \t %.4f" % (errors_lstm[
        #       0], errors_lstm[1], errors_lstm[2], errors_lstm[3]))
        # print("GRU_RNN:\t %.4f \t %.4f \t %.4f \t %.4f" % (errors_gru[
        #       0], errors_gru[1], errors_gru[2], errors_gru[3]))

        # TODO: Comment for now
        # rate = [1.0 * 1 / 10, 1.0 * 3 / 10, 1.0 *
        #         5 / 10, 1.0 * 7 / 10, 1.0 * 9 / 10]
        # errors_sarima = []
        # errors_lstm = []
        # errors_gru = []

        # for r in range(len(rate)):
        #     # print r
        #     errors_sarima.append(predict_sarima(number=(i + 1),
        #                                         index=x,
        #                                         data=y,
        #                                         ratio=rate[r])[0])

        #     lstm, gru, a = predict_RNN(number=(i + 1),
        #                                index=x,
        #                                data=y,
        #                                ratio=rate[r])
        #     # print(errors_sarima[r], lstm, gru)
        #     errors_lstm.append(lstm)
        #     errors_gru.append(gru)

        # print("********************")
        # print("Test Score (R2)")
        # print("====================")
        # print("portion \t SARIMA \t\t LSTM_RNN \t\t GRU_RNN")
        # for r in range(len(rate)):
        #     print("%s \t\t %.4f \t\t %.4f \t\t %.4f" %
        #           (rate[r], errors_sarima[r][2], errors_lstm[r][2], errors_gru[r][2]))
        plt.show()
    else:
        print("There is no such kind of file")


def getMinFileLen(files):
    data_len = 0
    file_num = len(files)
    for i in xrange(file_num):
        current_data_len = len(np.loadtxt(files[i], delimiter=" "))
        if data_len == 0:
            data_len = current_data_len
        else:
            data_len = data_len if data_len <= current_data_len else current_data_len
    return data_len


def create_dataset(dataset, look_back=1):
    '''
    Convert an array of values into a dataset matrix
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def format_dataset(dataset, seq_len, len_ratio):

    sequence_length = seq_len + 1
    result = []
    for index in range(len(dataset) - sequence_length):
        result.append(dataset[index:(index + sequence_length)])

    result = np.array(result)

    row = round(len_ratio * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def build_lstm_model(layers, dropout_ratio=0.2):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        units=layers[1],
        return_sequences=True))
    model.add(Dropout(dropout_ratio))

    if len(layers) >= 4:
        model.add(LSTM(
            layers[2],
            return_sequences=False))
        model.add(Dropout(dropout_ratio))
        model.add(Dense(units=layers[3]))
    else:
        model.add(Dense(units=layers[2]))

    model.add(Activation("tanh"))

    start = time.time()
    model.compile(loss="mse", optimizer="adam")
    print("> Compilation Time: ", time.time() - start)
    return model


def build_gru_model(layers, dropout_ratio=0.2):
    model = Sequential()

    model.add(GRU(
        input_shape=(layers[1], layers[0]),
        units=layers[1],
        return_sequences=True))
    model.add(Dropout(dropout_ratio))

    if len(layers) >= 4:
        model.add(GRU(
            layers[2],
            return_sequences=False))
        model.add(Dropout(dropout_ratio))
        model.add(Dense(units=layers[3]))
    else:
        model.add(Dense(units=layers[2]))

    model.add(Activation("tanh"))

    start = time.time()
    model.compile(loss="mse", optimizer="adam")
    print("> Compilation Time: ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    predict = model.predict(data)
    # predict = np.reshape(predict, (predict.size,))
    return predict


def predict_sequence_full(model, data, window_size):
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(
            curr_frame, [window_size - 1], predicted[-1], axis=0)
    return np.array(predicted)


def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(
                curr_frame, [window_size - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
    return prediction_seqs


def mean_absolute_percentage_error(val_actual, val_predict):
    '''
    Extend the MAPE function
    '''
    # val_actual, val_predict = check_array(val_actual, val_predict)
    return np.mean(1.0 * np.abs((val_actual - val_predict) / val_actual)) * 100


def predict_rnn_section(number, index, data, ratio):
    print("###################################")
    print("predict_rnn_section")
    sValue = 48
    interval = 30

    np.random.seed(7)
    section_number = int(len(data) / sValue)
    print("residual: ")
    print data

    df = pd.DataFrame({'year': 1999,
                       'month': 3,
                       'day': 1,
                       'minute': index * interval})
    data = pd.Series(data, index=pd.to_datetime(df))

    datasetList = []
    for i in range(sValue):
        tempList = []
        for j in range(section_number):
            tempList.append(data[i + j * sValue])
        datasetList.append(np.array(tempList).reshape(-1, 1))
    print("datasetList: ")
    print datasetList

    print("> Formatting data...")
    seq_len = 2
    batch_size = 512
    epochs = 10000
    layers_structure = [1, seq_len, 4, 1]
    trainX_List = []
    trainY_List = []
    testX_List = []
    testY_List = []
    for i in range(len(datasetList)):
        trainX, trainY, testX, testY = format_dataset(datasetList[i], 
            seq_len=seq_len, len_ratio=ratio)
        trainX_List.append(trainX)
        trainY_List.append(trainY)
        testX_List.append(testX)
        testY_List.append(testY)
    print("> Format completed.")

    # Create and fit the LSTM network
    trainPredict_List = []
    testPredict_List = []
    for i in range(len(datasetList)):      
        model = build_lstm_model(layers_structure)  
        model.fit(
            trainX_List[i],
            trainY_List[i],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.05,
            verbose=0)

        # Make predictions
        trainPredict_List.append(
            float(predict_point_by_point(model, trainX_List[i])[0])) 
        testPredict_List.append(
            float(predict_point_by_point(model, testX_List[i])[0]))
    
    print testPredict_List
    print len(testPredict_List)

    return testPredict_List


## Test ues
def predict_improved_lsarima2(number, index, data, ratio):

    # Seasonal value, each 48-example is a loop
    sValue = 48
    # 30 minutes
    interval = 30

    np.random.seed(7)
    dataset = data.reshape(-1, 1)
    dataset = normalized(dataset)

    df = pd.DataFrame({'year': 1999,
                       'month': 3,
                       'day': 1,
                       'minute': index * interval})
    norm_data = normalized(data)
    print norm_data
    data = pd.Series(norm_data, index=pd.to_datetime(df))

    # Build models
    models = []

    model = sm.tsa.statespace.SARIMAX(data.values,
                                      trend='n',
                                      order=(0, 1, 0),
                                      seasonal_order=(1, 1, 1, sValue))
    results = model.fit(disp=0)
    decompose_freq = 6
    decomposition = seasonal_decompose(data.values,
                                       model="additive",
                                       freq=decompose_freq)
    residual = decomposition.resid

    count = 0
    for i in range(len(residual)):
        if np.isnan(residual[i]):
            residual[i] = 0.0
    print("### Residuals ###")
    print len(residual)
    print len(data)
    print residual
    print count
    print("#################")

    # Predict the residual
    # Reshape into X=t and Y=t+look_back
    forecastResidual = residual.reshape(-1, 1)
    forecastResidual[:decompose_freq / 2] = 0
    forecastResidual[-decompose_freq / 2:] = 0

    size = int(len(residual) * ratio)
    forcast_ind = 48
    seq_len = 12
    # print("> Formatting data...")
    # train_residualX, train_residualY, test_residualX, test_residualY = format_dataset(
    #     forecastResidual, seq_len, len_ratio=ratio)
    # print("> Format completed.")

    # # Create and fit the LSTM network
    # look_back = 1
    # batch_size = 512
    # epochs = 100
    # layers_structure = [1, seq_len, 100, 1]

    # model_residual = build_lstm_model(layers=layers_structure)

    # model_residual.fit(
    #     train_residualX,
    #     train_residualY,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     verbose=2)

    # Make predictions
    # predict_forecastResidual = predict_point_by_point(
    #     model_residual, test_residualX)
    # ================+
    # predict_forecastResidual = predict_sequence_full(
    #     model=model_residual,
    #     data=test_residualX,
    #     window_size=seq_len)
    # =================
    predict_forecastResidual = predict_rnn_section(number=number, 
        index=index, data=residual, ratio=ratio)

    print("====predict_forecastResidual====")
    print len(predict_forecastResidual)
    print predict_forecastResidual
    print("===================")

    predict_begin = int(ratio * len(index))
    predict_end = int(len(index))
    data_forecast = results.predict(start=predict_begin,
                                    end=predict_end,
                                    dynamic=True)
    data_forecast_with_residual_sarima = copy.deepcopy(data_forecast)
    data_forecast_with_residual_lsarima = copy.deepcopy(data_forecast)

    print("=====data_forecast_with_residual=====")
    print len(data_forecast_with_residual_sarima)
    print data_forecast_with_residual_sarima
    print("===================")

    #====================================
    # Residual calculated by sarima
    for i in range(len(predict_forecastResidual)):
        data_forecast_with_residual_sarima[
            i] += forecastResidual[i - len(predict_forecastResidual)]
    #====================================

    #====================================
    # Residual calcuated by lsarima
    # forecast_ind = -24
    for i in range(len(predict_forecastResidual)):
        data_forecast_with_residual_lsarima[
            i] += predict_forecastResidual[i - len(predict_forecastResidual)]
    # print len(data_forecast)
    #====================================
    # Plot prediction results
    duration = pd.to_datetime(
        df)[predict_begin:predict_end-1]
    print len(duration)
    # print len(data_forecast) == len(duration)
    data_forecast = pd.Series(data_forecast[0:-2], index=duration)
    data_forecast_with_residual_sarima = pd.Series(
        data_forecast_with_residual_sarima[0:-2], index=duration)
    data_forecast_with_residual_lsarima = pd.Series(
        data_forecast_with_residual_lsarima[0:-2], index=duration)

    # ==============================
    # print residual prediction
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    residual = pd.Series(residual, index=pd.to_datetime(df))
    ax1 = residual.ix['1999-03-01 00:00:00':].plot(ax=ax1)
    predict_forecastResidual = pd.Series(predict_forecastResidual[0:-1], index=duration)
    predict_forecastResidual.plot(ax=ax1, color='black')
    # ==============================

    # print predictions
    fig, ax = plt.subplots(figsize=(10, 3))
    ax = data.ix['1999-03-01 00:00:00':].plot(ax=ax)
    # Figure 8
    data_forecast.plot(ax=ax, color='red')
    data_forecast_with_residual_sarima.plot(ax=ax, color='black')
    data_forecast_with_residual_lsarima.plot(ax=ax, color='green')

    # print MSE
    testScore_mse = mean_squared_error(
        data[predict_begin:-1], data_forecast_with_residual_lsarima)

    # print MAE
    testScore_mae = mean_absolute_error(
        data[predict_begin:-1], data_forecast_with_residual_lsarima)

    # print MAPE
    testScore_mape = mean_absolute_percentage_error(
        data[predict_begin:-1], data_forecast_with_residual_lsarima)

    # print R squre
    testScore_r2 = r2_score(
        data[predict_begin:-1], data_forecast_with_residual_lsarima)

    error = [testScore_mse, testScore_mae, testScore_mape, testScore_r2]

    return (error, data_forecast_with_residual_lsarima)

###########


def predict_improved_lsarima(number, index, data, ratio):

    # Seasonal value, each 48-example is a loop
    sValue = 48
    # 30 minutes
    interval = 30

    np.random.seed(7)
    dataset = data.reshape(-1, 1)
    dataset = normalized(dataset)

    df = pd.DataFrame({'year': 1999,
                       'month': 3,
                       'day': 1,
                       'minute': index * interval})
    norm_data = normalized(data)

    data = pd.Series(norm_data, index=pd.to_datetime(df))

    # Build models
    models = []

    # seasonal_order=(P,D,Q,s),
    # s is the sequence, representing the number of examples in a period
    # (1,1,1)(1,1,1,48)
    model = sm.tsa.statespace.SARIMAX(data.values,
                                      trend='n',
                                      order=(0, 1, 0),
                                      seasonal_order=(1, 1, 1, sValue))
    results = model.fit(disp=0)
    # print results.summary()
    decompose_freq = 6
    decomposition = seasonal_decompose(data.values,
                                       model="additive",
                                       freq=decompose_freq)
    residual = decomposition.resid

    count = 0
    for i in range(len(residual)):
        if not np.isnan(residual[i]):
            count += 1
    enter = "\r\n"
    logfile = "test.txt"
    print("### Residuals ###")
    print len(residual)
    print residual
    print count
    print("#################")

    # Predict the residual
    # Reshape into X=t and Y=t+look_back
    forecastResidual = residual.reshape(-1, 1)
    forecastResidual[:decompose_freq / 2] = 0
    forecastResidual[-decompose_freq / 2:] = 0

    size = int(len(residual) * ratio)
    forcast_ind = 48
    seq_len = 48
    # testforecastResidual = forecastResidual[forcast_ind:-forcast_ind]
    print("> Formatting data...")
    train_residualX, train_residualY, test_residualX, test_residualY = format_dataset(
        forecastResidual, seq_len, len_ratio=ratio)
    print("> Format completed.")

    print("====== test_residualX ======")  
    print test_residualX
    print("============================")
    print("====== test_residualY ======")
    print test_residualY
    print("============================")

    # Create and fit the LSTM network
    look_back = 1
    batch_size = 512
    epochs = 1000
    layers_structure = [1, seq_len, 10, 1]

    model_residual = build_lstm_model(layers=layers_structure)

    model_residual.fit(
        train_residualX,
        train_residualY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2)

    # Make predictions
    # predict_forecastResidual = predict_point_by_point(
    #     model_residual, test_residualX)
    predict_forecastResidual = predict_sequence_full(
        model=model_residual,
        data=test_residualX,
        window_size=seq_len)

    print("====predict_forecastResidual====")
    print len(predict_forecastResidual)
    print predict_forecastResidual
    print("===================")

    predict_begin = int(ratio * len(index))
    predict_end = int(len(index))
    data_forecast = results.predict(start=predict_begin,
                                    end=predict_end,
                                    dynamic=True)
    data_forecast_with_residual = copy.deepcopy(data_forecast)

    print("=====data_forecast_with_residual=====")
    print len(data_forecast_with_residual)
    print data_forecast_with_residual
    print("===================")
    # print predict_forecastResidual.shape
    # for i in range(len(predict_forecastResidual)):
    #     data_forecast_with_residual[i] += predict_forecastResidual[i]

    #====================================
    # Residual calculated by sarima
    # for i in range(len(predict_forecastResidual)):
    #     data_forecast_with_residual[
    #         i] += forecastResidual[i - len(predict_forecastResidual)]
    #====================================

    #====================================
    # Residual calcuated by lsarima
    # forecast_ind = -24
    for i in range(len(predict_forecastResidual)):
        data_forecast_with_residual[
            i] += predict_forecastResidual[i - len(predict_forecastResidual)]
    print len(data_forecast)
    #====================================
    # Plot prediction results
    duration = pd.to_datetime(
        df)[predict_begin:predict_end - 1]
    print len(duration)
    # print len(data_forecast) == len(duration)
    data_forecast = pd.Series(data_forecast[0:-2], index=duration)
    data_forecast_with_residual = pd.Series(
        data_forecast_with_residual[0:-2], index=duration)

    # #==============================================
    # # plot residuals
    # print forecastResidual.shape
    # print len(forecastResidual.ravel())
    # print predict_forecastResidual.shape
    # print len(predict_forecastResidual.ravel())
    # fig, ax = plt.subplots(figsize=(10, 3))
    # forecastResidual = pd.Series(forecastResidual.ravel(), index=pd.to_datetime(df))
    # if (len(duration) >= len(predict_forecastResidual.ravel())):        
    #     predict_forecastResidual = pd.Series(predict_forecastResidual.ravel(), index=duration[0:-(len(duration)-len(predict_forecastResidual.ravel()))])
    # else:
    #     predict_forecastResidual = pd.Series(predict_forecastResidual.ravel()[0:-(len(predict_forecastResidual.ravel())-len(duration))], index=duration)
    
    # print("=====forecastResidual changed=====")
    # print forecastResidual
    # print("=================")

    # print("=====predict_forecastResidual changed=====")
    # print predict_forecastResidual
    # print("=================")

    # print("=====data=====")
    # print data
    # print("=================")

    # ax = forecastResidual.ix['1999-03-01 00:00:00':].plot(ax=ax, color='black', label='Original')
    # predict_forecastResidual.plot(ax=ax, color='red', label='Predicted')
    # # plt.plot(forecastResidual, color='black', label='Original')
    # # plt.plot(predict_forecastResidual, color='red', label='Predicted')
    # plt.legend(loc='best')
    # plt.title('Residuals')
    # plt.show
    # #==============================================

    # print predictions
    fig, ax = plt.subplots(figsize=(10, 3))
    ax = data.ix['1999-03-01 00:00:00':].plot(ax=ax)
    # Figure 8
    data_forecast.plot(ax=ax, color='red')
    data_forecast_with_residual.plot(ax=ax, color='black')

    # #==================================================
    # # Test purpose
    # # print MSE
    # testScore_mse = mean_squared_error(
    # data[predict_begin - 1:-2 + forecast_ind],
    # data_forecast_with_residual[:forecast_ind])

    # # print MAE
    # testScore_mae = mean_absolute_error(
    # data[predict_begin - 1:-2 + forecast_ind],
    # data_forecast_with_residual[:forecast_ind])

    # # print MAPE
    # testScore_mape = mean_absolute_percentage_error(
    # data[predict_begin - 1:-2 + forecast_ind],
    # data_forecast_with_residual[:forecast_ind])

    # # print R squre
    # testScore_r2 = r2_score(
    #     data[predict_begin - 1:-2 + forecast_ind], data_forecast_with_residual[:forecast_ind])
    # #==================================================

    # print MSE
    testScore_mse = mean_squared_error(
        data[predict_begin:-1], data_forecast_with_residual)

    # print MAE
    testScore_mae = mean_absolute_error(
        data[predict_begin:-1], data_forecast_with_residual)

    # print MAPE
    testScore_mape = mean_absolute_percentage_error(
        data[predict_begin:-1], data_forecast_with_residual)

    # print R squre
    testScore_r2 = r2_score(
        data[predict_begin:-1], data_forecast_with_residual)

    error = [testScore_mse, testScore_mae, testScore_mape, testScore_r2]

    return (error, data_forecast_with_residual)

    #=======================================
    # # print MSE
    # testScore_mse = mean_squared_error(data[predict_begin - 1:], data_forecast)
    # # # print("Test Score: %.4f MSE" % error)
    # # testResidual = data[predict_begin - 1:] - data_forecast
    # # print("## testResidual ##")
    # # print testResidual
    # # print("##################")

    # # print MAE
    # testScore_mae = mean_absolute_error(
    #     data[predict_begin - 1:], data_forecast)

    # # print MAPE
    # testScore_mape = mean_absolute_percentage_error(
    #     data[predict_begin - 1:], data_forecast)

    # # print R squre
    # testScore_r2 = r2_score(data[predict_begin - 1:], data_forecast)

    # error = [testScore_mse, testScore_mae, testScore_mape, testScore_r2]

    # return (error, data_forecast)
    #=======================================


def predict_RNN(number, index, data, ratio):
    '''
    This uses Recurrent Neural Networks to predict network traffic, including
    Long Short-Term Memory(LSTM) and Gated Recurrent Units(GRU)
    '''

    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load the dataset
    dataset = data.reshape(-1, 1)
    # print data
    # print dataset

    # Normalized the dataset
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(dataset)
    dataset = normalized(dataset)

    # Format the dataset
    size = int(len(dataset) * ratio)
    seq_len = 5
    look_back = 1
    batch_size = 512
    epochs = 1000
    layers_structure = [1, seq_len, 10, 1]
    print("> Formatting data...")
    trainX, trainY, testX, testY = format_dataset(
        dataset, seq_len, len_ratio=ratio)
    print(trainY.shape)
    print("> Format completed.")

    # Create and fit the LSTM network
    model = build_lstm_model(layers_structure)
    model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        verbose=2)

    # Create and fit the GRU network
    model2 = build_gru_model(layers_structure)
    model2.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        verbose=2)

    # Make predictions
    trainPredict = predict_point_by_point(model, trainX)
    testPredict = predict_point_by_point(model, testX)
    # testPredict = predict_sequence_full(model=model,
    #                                     data=testX,
    #                                     window_size=seq_len)
    trainPredict2 = predict_point_by_point(model2, trainX)
    testPredict2 = predict_point_by_point(model2, testX)
    # testPredict2 = predict_sequence_full(model=model2,
    #                                     data=testX,
    #                                     window_size=seq_len)

    # Invert predictions
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform([trainY])
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform([testY])

    # print("trainY")
    # print trainY
    # print("===========================")
    # print("trainPredict")
    # print trainPredict[:, 0]
    # print("===========================")

    # # Calculate root mean squared error
    # # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    # trainScore = mean_squared_error(trainY, trainPredict)**0.5
    # print("Train Score: %.2f RMSE" % (trainScore))
    # # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    # testScore = mean_squared_error(testY, testPredict)**0.5
    # print("Test Score: %.2f RMSE" % (testScore))

    # Calculate mean squared error
    trainScore = mean_squared_error(trainY, trainPredict)
    # print("Train Score: %.4f MSE" % (trainScore))
    testScore = mean_squared_error(testY, testPredict)
    # print("Test Score: %.4f MSE" % (testScore))
    trainScore2 = mean_squared_error(trainY, trainPredict2)
    # print("Train Score: %.4f MSE" % (trainScore))
    testScore2 = mean_squared_error(testY, testPredict2)
    # print("Test Score: %.4f MSE" % (testScore))

    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict
    trainPredictPlot2 = np.empty_like(dataset)
    trainPredictPlot2[:] = np.nan
    trainPredictPlot2[look_back:len(trainPredict) + look_back] = trainPredict2

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    testPredictPlot2 = np.empty_like(dataset)
    testPredictPlot2[:] = np.nan
    # testPredictPlot[len(trainPredict) + (look_back * 2) +
    #                 1:len(dataset) - 1] = testPredict
    # print len(testPredict)
    # print len(dataset) - size
    size += len(testPredictPlot[size:len(dataset) - 2]) - len(testPredict)
    testPredictPlot[size:len(dataset) - 2] = testPredict
    testPredictPlot2[size:len(dataset) - 2] = testPredict2

    # Plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    fig = plt.figure(figsize=(10, 3))
    plt.plot(dataset, color='blue', label='Original')
    plt.plot(testPredictPlot2, color='red', label='GRU')
    plt.plot(testPredictPlot, color='black', label='LSTM')
    plt.legend(loc='best')
    plt.title('Prediction by RNN')
    # plt.plot(dataset)
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)

    # calculate MSE
    testScore_mse = mean_squared_error(
        dataset[size:len(dataset) - 2], testPredict)
    testScore2_mse = mean_squared_error(
        dataset[size:len(dataset) - 2], testPredict2)

    # calculate MAE
    testScore_mae = mean_absolute_error(
        dataset[size:len(dataset) - 2], testPredict)
    testScore2_mae = mean_absolute_error(
        dataset[size:len(dataset) - 2], testPredict2)

    # calculate r square
    testScore_r2 = r2_score(
        dataset[size:len(dataset) - 2], testPredict)
    testScore2_r2 = r2_score(
        dataset[size:len(dataset) - 2], testPredict2)

    # calculate MAPE
    testScore_mape = mean_absolute_percentage_error(
        dataset[size:len(dataset) - 2], testPredict)
    testScore2_mape = mean_absolute_percentage_error(
        dataset[size:len(dataset) - 2], testPredict2)

    testScore_err = [testScore_mse, testScore_mae,
                     testScore_mape, testScore_r2]
    testScore2_err = [testScore2_mse, testScore2_mae,
                      testScore_mape, testScore2_r2]

    return (testScore_err, testScore2_err, testPredictPlot)


def predict_LSTM_RNN(number, index, data, ratio):
    '''
    This uses Long Short-Term Memory Recurrent Neural Networks
    '''

    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load the dataset
    dataset = data.reshape(-1, 1)

    dataset = normalized(dataset)

    # Format the dataset
    size = int(len(dataset) * ratio)
    seq_len = 48
    print("> Formatting data...")
    trainX, trainY, testX, testY = format_dataset(
        dataset, seq_len, len_ratio=ratio)
    print(trainY.shape)
    print("> Format completed.")

    # Create and fit the LSTM network
    look_back = 1
    batch_size = 512
    epochs = 1000
    model = build_lstm_model([1, seq_len, 100, 1])
    # model = build_lstm_model([1, seq_len, 1])

    model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        verbose=2)

    # Make predictions
    # size = size + 1
    trainPredict = predict_point_by_point(model, trainX)
    # testPredict = predict_point_by_point(model, testX)
    # testPredict = predict_sequences_multiple(model=model,
    #                                          data=testX,
    #                                          window_size=seq_len,
    #                                          prediction_len=48)
    testPredict = predict_sequence_full(model=model,
                                        data=testX,
                                        window_size=seq_len)

    # Calculate mean squared error
    trainScore = mean_squared_error(trainY, trainPredict)
    print("Train Score: %.4f MSE" % (trainScore))

    # # For point to point
    # testScore = mean_squared_error(testY, testPredict)
    # print("Test Score: %.4f MSE" % (testScore))

    # # For multiple points
    # print testPredict
    # testPredict = np.reshape(np.asarray(testPredict[0]), (-1, 1))
    # testScore = mean_squared_error(testY, testPredict)
    # print("Test Score: %.4f MSE" % (testScore))

    # For sequence full
    testPredict = np.reshape(np.asarray(testPredict), (-1, 1))
    testScore = mean_squared_error(testY, testPredict)
    print("Test Score: %.4f MSE" % (testScore))

    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back: len(trainPredict) + look_back] = trainPredict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan

    print(testPredict.shape)
    size += len(testPredictPlot[size: len(dataset) - 2]) - len(testPredict)
    testPredictPlot[size: len(dataset) - 2] = testPredict

    # Plot baseline and predictions
    fig = plt.figure(figsize=(10, 3))
    plt.plot(dataset, color='blue', label='Original')
    plt.plot(testPredictPlot, color='black', label='testPredictPlot')
    plt.legend(loc='best')
    plt.title('Prediction by LSTM NN')

    testScore1 = mean_squared_error(
        dataset[size:len(dataset) - 2], testPredict)

    return (testScore1, testPredictPlot)


def test_stationarity(timeseries):

    # determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # plot rolling statistics
    fig = plt.figure(figsize=(13, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    # Perform Dickey-Fuller test
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    # 'p-value' is a coefficient, where a unit root is present if p = 1
    # Links - https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test
    dfoutput = pd.Series(dftest[0:4], index=[
        'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


def predict_sarima(number, index, data, ratio):

    # Seasonal value, each 48-example is a loop
    sValue = 48
    # 30 minutes
    interval = 30
    # print sValue
    # print interval
    df = pd.DataFrame({'year': 1999,
                       'month': 3,
                       'day': 1,
                       'minute': index * interval})
    norm_data = normalized(data)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # norm_data = scaler.fit_transform(data)
    data = pd.Series(norm_data, index=pd.to_datetime(df))
    # print data
    # print("############################")

    # TODO: Comment for now
    # # Test the stationary of data
    # # Figure 2
    # test_stationarity(data)

    # decomposition = seasonal_decompose(data.values, freq=sValue)
    # # Figure 3
    # fig = decomposition.plot()
    # fig.set_size_inches(13, 8)

    # # Test the stationary of log(data)
    # log_data = map(lambda x: np.log(x), data.values)
    # log_data = pd.Series(log_data, index=pd.to_datetime(df))
    # # Figure 4
    # test_stationarity(log_data)

    # # Check first difference
    # order = 1
    # data_first_diff = data.diff(order)
    # # for i in range(order):
    # #     data_first_diff[i] = 0.0
    # change2zero(data_first_diff, order)
    # # print data_first_diff
    # # Figure 5
    # test_stationarity(data_first_diff)

    # # Check seasonal first difference
    # order = sValue
    # data_seasonal_first_diff = data_first_diff - data_first_diff.shift(order)
    # # for i in range(order):
    # #     data_seasonal_first_diff[i] = 0.0
    # change2zero(data_seasonal_first_diff, order)
    # # print data_seasonal_first_diff
    # # Figure 6
    # test_stationarity(data_seasonal_first_diff)

    # # Check log seasonal first difference
    # # log_data_first_diff = log_data.diff(1)
    # # log_data_seasonal_first_diff = log_data_first_diff - log_data_first_diff.shift(sValue)
    # # change2zero(log_data_seasonal_first_diff, order+1)
    # # print log_data_seasonal_first_diff
    # # test_stationarity(log_data_seasonal_first_diff)

    # # Plot autocorrelation and partial autocorrelation
    # # Figure 7
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(
    #     data_seasonal_first_diff, lags=40, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(
    #     data_seasonal_first_diff, lags=40, ax=ax2)

    # Build models
    models = []

    # Find the most suitable parameters - version 2
    # ====================================
    # para = 6
    # aic = pd.DataFrame(np.zeros((para, para), dtype=float))
    # for p in range(para):
    #     for q in range(para):
    #         mod = sm.tsa.statespace.SARIMAX(data.values,
    #                                         trend='n',
    #                                         order=(p, 1, q),
    #                                         seasonal_order=(
    #                                             1, 1, 1, sValue),
    #                                         enforce_invertibility=False)
    #         try:
    #             res = mod.fit(disp=False)
    #             aic.iloc[p, q] = res.aic
    #         except:
    #             aic.iloc[p, q] = np.nan
    # print aic
    # ====================================

    # # Find the most suitable parameters - version 1
    # # ====================================
    # para = 3
    # parameter_dict = {}
    # count = 0
    # for p in range(para):
    #     for q in range(para - 1):
    #         for P in range(para):
    #             for Q in range(1, para):
    #                 parameter_dict['%d' % count] = (p, q, P, Q)
    #                 models.append(sm.tsa.statespace.SARIMAX(data.values,
    #                                                         trend='n',
    #                                                         order=(p, 1, q),
    #                                                         seasonal_order=(P, 1, Q, sValue),
    #                                                         enforce_invertibility=False))
    #                 count += 1

    # predict_begin = int(0.90 * len(index))
    # predict_end = int(len(index))
    # duration = pd.to_datetime(df)[predict_begin - 1:predict_end]
    # exclude = [1, 7, 13, 19, 25, 31]
    # error_list = []
    # for i in range(len(models)):
    #     if i in exclude:
    #         continue
    #     print i
    #     results = models[i].fit(disp=0)
    #     # print results.summary()

    #     # Make predictions
    #     data_forecast = results.predict(start=predict_begin,
    #                                     end=predict_end,
    #                                     dynamic=True)

    #     # print len(data_forecast)
    #     # Plot prediction results

    #     # print len(duration)
    #     # print len(data_forecast) == len(duration)
    #     data_forecast = pd.Series(data_forecast, index=duration)

    #     # print predictions
    #     fig, ax = plt.subplots(figsize=(10, 3))
    #     ax = data.ix['1999-03-01 00:00:00':].plot(ax=ax)
    #     # Figure 8
    #     data_forecast.plot(ax=ax, color='red')

    #     # print MSE
    #     error = mean_squared_error(data[predict_begin - 1:], data_forecast)
    #     error_list.append([i, error])
    #     # print error

    # # print error_list
    # ind = sorted(error_list, key=getKey)[0]
    # print ind
    # print ind[0]
    # print parameter_dict['%d' % ind[0]]
    # # ====================================

    # Comment for now
    # seasonal_order=(P,D,Q,s),
    # s is the sequence, representing the number of examples in a period
    # (1,1,1)(1,1,1,48)
    model = sm.tsa.statespace.SARIMAX(data.values,
                                      trend='n',
                                      order=(0, 1, 0),
                                      seasonal_order=(1, 1, 1, sValue))
    results = model.fit(disp=0)
    # print results.summary()

    # Make predictions
    predict_begin = int(ratio * len(index))
    predict_end = int(len(index))
    data_forecast = results.predict(start=predict_begin,
                                    end=predict_end,
                                    dynamic=True)

    # print len(data_forecast)
    # Plot prediction results
    duration = pd.to_datetime(df)[predict_begin - 1:predict_end]
    # print len(duration)
    # print len(data_forecast) == len(duration)
    data_forecast = pd.Series(data_forecast, index=duration)

    # print predictions
    fig, ax = plt.subplots(figsize=(10, 3))
    ax = data.ix['1999-03-01 00:00:00':].plot(ax=ax)
    # Figure 8
    data_forecast.plot(ax=ax, color='red')

    # print MSE
    testScore_mse = mean_squared_error(data[predict_begin - 1:], data_forecast)
    # print("Test Score: %.4f MSE" % error)

    # print MAE
    testScore_mae = mean_absolute_error(
        data[predict_begin - 1:], data_forecast)

    # print MAPE
    testScore_mape = mean_absolute_percentage_error(
        data[predict_begin - 1:], data_forecast)

    # print R squre
    testScore_r2 = r2_score(data[predict_begin - 1:], data_forecast)

    error = [testScore_mse, testScore_mae, testScore_mape, testScore_r2]

    return (error, data_forecast)


def getKey(item):
    '''Sort the list based on the second sub-list'''
    return item[1]


def change2zero(data, order):
    '''Change the NaN to 0.0'''
    for i in range(order):
        data[i] = 0.0


def normalized1(x):
    return 0.8 * (x - min(x)) / (max(x) - min(x)) + 0.1


def normalized(x):
    return (x - min(x)) / (max(x) - min(x))


def predict_arima(number, index, data):
    # Set the index as date type
    # print number
    df = pd.DataFrame({'year': 1999,
                       'month': 3,
                       'day': 1,
                       'minute': index * 5})
    # print df
    # print pd.to_datetime(df)
    # normalize data
    # norm_data = normalize(data, axis=1) + 0.5
    norm_data = normalized(data)
    # print norm_data
    data = pd.Series(norm_data, index=pd.to_datetime(df))
    X = data.values
    # print X
    size = int(len(X) * 0.8)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions1 = list()
    predictions2 = list()
    predictions3 = list()

    i, j = 2, 2
    res = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='nc')
    # print res.aic_min_order
    # print res.bic_min_order
    for t in range(len(test)):
        # model = ARIMA(history, order=(i, 1, j))
        # model = ARIMA(history, order=(res.aic_min_order[0],
        #                               1,
        #                               res.aic_min_order[1]))
        # ARIMA - Min AIC
        model1 = ARIMA(history, order=(
            res.aic_min_order[0], 1, res.aic_min_order[1]))
        # ARIMA - Min BIC
        model2 = ARIMA(history, order=(
            res.bic_min_order[0], 1, res.bic_min_order[1]))

        model_fit1 = model1.fit(disp=0)
        model_fit2 = model2.fit(disp=0)

        output1 = model_fit1.forecast()
        output2 = model_fit2.forecast()

        # print output
        yhat1 = output1[0][0]
        yhat2 = output2[0][0]

        predictions1.append(yhat1)
        predictions2.append(yhat2)

        obs = test[t]

        history.append(obs)

        # print('predicted=%f, expected=%f' % (yhat, obs))
    error1 = mean_squared_error(test, predictions1)
    error2 = mean_squared_error(test, predictions2)
    print('Test MSE of ARIMA - Min AIC: %.3f' % error1)
    print('Test MSE of ARIMA - Min BIC: %.3f' % error2)

    # plot
    # duration = pd.to_datetime(df)[size - 1:len(X) - 1]
    duration = pd.to_datetime(df)[size:len(X)]
    predictions1 = pd.Series(predictions1, index=duration)
    predictions2 = pd.Series(predictions2, index=duration)

    # print predictions
    fig, ax = plt.subplots(figsize=(10, 3))
    ax = data.ix['1999-03-01 00:00:00':].plot(ax=ax)
    predictions1.plot(ax=ax, color='red')
    predictions2.plot(ax=ax, color='green')


def predict_arma(number, index, data, original_index, original_data):
    # axes list
    ax = []
    # difference list
    diff = []
    # order number
    order_num = 5
    # Set the index as date type
    df = pd.DataFrame({'year': 1999,
                       'month': 3,
                       'day': 1,
                       'minute': index * 5})
    original_df = pd.DataFrame({'year': 1999,
                                'month': 3,
                                'day': 1,
                                'minute': original_index * 5})
    # print pd.to_datetime(df)
    data = pd.Series(data, index=pd.to_datetime(df))
    original_data = pd.Series(original_data, index=pd.to_datetime(original_df))
    # data = pd.Series(data, index=pd.to_datetime(index, unit='m'))
    # print data

    fig = plt.figure("Differences of Diverse Orders in Day %s" %
                     number, figsize=(10, 4 * order_num))

    # Show differences with i-order
    # for i in range(1, order_num + 1):
    #     ax.append(fig.add_subplot(order_num, 1, i))
    #     # Get the difference of time series, which is
    #     # the d parameter of ARIMA(p, d, q)
    #     diff.append(data.diff(i))
    #     # Plot the i-order of difference
    #     diff[i - 1].plot(ax=ax[i - 1])

    # # After observation, choose first-order difference
    order = 1
    data = data.diff(order)
    original_data = original_data.diff(order)
    print data
    print original_data
    # "data[0]=NaN" causes the autocorrelation figure shows abnormally
    for i in range(order):
        data[i] = 0.0
        original_data[i] = 0.0
    # autocorrelation_plot(data)
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)

    #==================================
    # arma_mod70 = sm.tsa.ARMA(data, (7, 0)).fit()

    arma_mod = []
    row = 3
    col = 3
    # ARMA(0, 2) is the best
    for i in range(row):
        temp = []
        for j in range(col):
            temp.append(sm.tsa.ARMA(data, (i, j)).fit())
        arma_mod.append(temp)

    for i in range(row):
        for j in range(col):
            print(arma_mod[i][j].aic, arma_mod[i][j].bic, arma_mod[i][j].hqic)
    #==================================

    # get the mininal value of aic/bic
    res = sm.tsa.arma_order_select_ic(data, ic=['aic', 'bic'], trend='nc')
    # print res.aic_min_order
    # print res.bic_min_order

    # arma_mod00 = sm.tsa.ARMA(data, (0, 0)).fit()
    # arma_mod01 = sm.tsa.ARMA(data, (0, 1)).fit()
    # arma_mod02 = sm.tsa.ARMA(data, (0, 2)).fit()
    # arma_mod10 = sm.tsa.ARMA(data, (1, 0)).fit()
    # arma_mod11 = sm.tsa.ARMA(data, (1, 1)).fit()
    # arma_mod12 = sm.tsa.ARMA(data, (1, 2)).fit()
    # arma_mod20 = sm.tsa.ARMA(data, (2, 0)).fit()
    # arma_mod21 = sm.tsa.ARMA(data, (2, 1)).fit()
    # arma_mod22 = sm.tsa.ARMA(data, (2, 2)).fit()

    # print(arma_mod00.aic, arma_mod00.bic, arma_mod00.hqic)
    # print(arma_mod01.aic, arma_mod01.bic, arma_mod01.hqic)
    # print(arma_mod02.aic, arma_mod02.bic, arma_mod02.hqic)
    # print(arma_mod10.aic, arma_mod10.bic, arma_mod10.hqic)
    # print(arma_mod11.aic, arma_mod11.bic, arma_mod11.hqic)
    # print(arma_mod12.aic, arma_mod12.bic, arma_mod12.hqic)
    # print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
    # print(arma_mod21.aic, arma_mod21.bic, arma_mod21.hqic)
    # print(arma_mod22.aic, arma_mod22.bic, arma_mod22.hqic)

    # Autocorrelation for ARMA(0, 2)
    # fit model
    # model = ARIMA(data_bak, order=(1, 1, 0))
    # model_fit = model.fit(disp=0)
    # print model_fit.summary()

    # # plot residual errors
    # residuals = DataFrame( model_fit.resid )
    # residuals.plot()
    # residuals.plot(kind='kde')
    # print residuals.describe()

    # row = 3
    # col = 3
    # model = []
    # model_fit = []

    # for i in range(row):
    #     temp = []
    #     for j in range(col):
    #         # print(i, j)
    #         temp.append(ARIMA(data_bak, order=(i, 1, j)))
    #     model.append(temp)

    # for i in range(row):
    #     temp = []
    #     for j in range(col):
    #         print(i, j)
    #         # print model[i][j].fit(disp=0).summary()
    #         temp.append(model[i][j].fit(disp=0))
    #     model_fit.append(temp)

    # for i in range(row):
    #     for j in range(col):
    #         print model_fit[i][j].summary()

    #         # plot residual errors
    #         residuals = DataFrame(model_fit[i][j].resid)
    #         residuals.plot()
    #         residuals.plot(kind='kde')
    #         print residuals.describe()

    # plot autocorrelation of residual errors
    # predict_model = arma_mod[0][2]
    predict_model = arma_mod[res.aic_min_order[0]][res.aic_min_order[1]]
    resid = predict_model.resid
    fig = plt.figure("Autocorrelation of residuals", figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

    # Durbin-Watson Exam
    # DW is in [0, 4], where
    # DW = 4 <=> p(rou) = -1, DW = 2 <=> p(rou) = 0, DW = 0 <=> p(rou) = 1
    print(sm.stats.durbin_watson(resid.values))

    # Check the data are from the same distribution or not
    fig = plt.figure("Check for the data validation", figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)

    # Ljung-Box Exam
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    lb_data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(lb_data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))

    # Prediction with arma model
    begin_time = str(pd.to_datetime(df)[len(index) - 1])
    end_time = str(pd.to_datetime(original_df)[len(original_index) - 1])
    # print pd.to_datetime(df)[len(index)-1]
    # print pd.to_datetime(original_df)[len(original_index)-1]
    predict_sunspots = predict_model.predict(begin_time,
                                             end_time,
                                             dynamic=True)
    print predict_sunspots
    fig, ax = plt.subplots(figsize=(10, 3))
    ax = original_data.ix['1999-03-01 00:00:00':].plot(ax=ax)
    predict_sunspots.plot(ax=ax)


def test():
    # fix random seed for reproducibility
    np.random.seed(7)

    # load the dataset
    dataframe = pd.read_csv('international-airline-passengers.csv',
                            usecols=[1],
                            engine='python',
                            skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    print("dataset")
    print dataset
    print("====================")

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    print("dataset")
    print dataset
    print("====================")

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("len(train), len(test)")
    print(len(train), len(test))
    print("====================")

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print("trainX")
    print(trainX)
    print("====================")

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print("trainX")
    print(trainX)
    print("====================")

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict
    print("trainPredictPlot")
    print(trainPredictPlot)
    print("====================")
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) +
                    1:len(dataset) - 1] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


def main():

    plotData()
    # test()


if __name__ == '__main__':
    main()
