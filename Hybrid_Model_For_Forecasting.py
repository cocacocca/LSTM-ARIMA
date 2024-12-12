import pandas as pd
import numpy as np
import os
import time

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import keras
from keras.src.optimizers import Adam
from matplotlib import font_manager
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


# 使用字体名称设置
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

load_dotenv(dotenv_path='.env')
Filename_address = os.getenv("FILE_ADDRESS")
Output_address = os.getenv("OUTPUT_ADDRESS")
close = "Close"
lag = int(os.getenv("LAG", 1))
epochs = int(os.getenv("EPOCHS", 10))
learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
batch_size = int(os.getenv("BATCH_SIZE", 32))
number_nodes = int(os.getenv("NUMBER_NODES", 50))
days = int(os.getenv("Prediction_days", 10))
n = int(os.getenv("NN_LAGS", 5))

# 加载数据并基于时间创建数据框
def data_loader():
   cols = ["Open", "High", "Low", "Close", "Volume"]
   data = pd.read_csv(Filename_address, index_col="Date", parse_dates=True)
   data.columns = cols
   data = data.dropna()
   print(f"数据集的形状是: {data.shape}\n数据集的前几行是:\n{data.head()}\n")
   return data


# 根据数据和列名绘制折线图
def plot_predictions(train, predictions,title):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='真实值')
    plt.plot(train.index, predictions, label='预测值', color='red')
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.savefig("预测.jpg")
    
def plot_raw_data(data):
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data[close], label='close')
    plt.title('原始时间序列数据')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.legend()
    plt.savefig('原始时间序列数据.jpg')


def plot_train_test(train, test):
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='train_set')
    plt.plot(test.index, test, label='test_set', color='orange')
    plt.title('训练集和测试集数据')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.savefig('训练集和测试集数据.jpg')


def plot_prediction_errors(errors):
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label='预测误差')
    plt.title('预测误差随时间变化')
    plt.xlabel('时间步长')
    plt.ylabel('误差')
    plt.legend()
    plt.savefig('预测误差随时间变化.jpg')


def plot_final_predictions(test, final_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test, label='actually')
    plt.plot(test.index, final_predictions, label='fix_pred', color='green')
    plt.title('修正后的最终预测结果')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.legend()
    plt.savefig('修正后的最终预测结果.jpg')


def plot_accuracy(mse, rmse, mae):
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'orange', 'green'])
    plt.title('模型准确性指标')
    plt.savefig('模型准确性指标.jpg')


def plot_arima_accuracy(mse, rmse, mae):
    metrics = ['均方误差(MSE)', '均方根误差(RMSE)', '平均绝对误差(MAE)']
    values = [mse, rmse, mae]
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'orange', 'green'])
    plt.title('ARIMA模型准确性指标')
    plt.savefig('ARIMA模型准确性指标.jpg')


# 数据划分用于模型的开发和训练。
def data_allocation(data):
    train_len_val = len(data) - days
    train, test = data[close].iloc[0:train_len_val], data[close].iloc[train_len_val:]
    print("\n--------------------------------- 训练集数据如下： -------------------------------------------\n")
    print(train)
    print(f"\n训练集的条目数 : {len(train)}\n")
    print("\n--------------------------------- 测试集数据如下： --------------------------------------------\n")
    print(test)
    print(f"\n测试集的条目数 : {len(test)}\n")
    return train, test


# 将数据转化为神经网络的滞后矩阵（n:矩阵）。
def apply_transform(data, n: int):
    middle_data = []
    target_data = []
    for i in range(n, len(data)):
        input_sequence = data[i - n:i]
        middle_data.append(input_sequence)
        target_data.append(data[i])
    middle_data = np.array(middle_data).reshape((len(middle_data), n, 1))
    target_data = np.array(target_data)
    return middle_data, target_data

# LSTM模型训练函数
def LSTM(train, n: int, number_nodes, learning_rate, epochs, batch_size):
    middle_data, target_data = apply_transform(train, n)
    model = keras.Sequential([
        keras.layers.Input((n, 1)),
        keras.layers.LSTM(number_nodes, input_shape=(n, 1)),
        keras.layers.Dense(units=number_nodes, activation="relu"),
        keras.layers.Dense(units=number_nodes, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=keras.src.optimizers.Adam(learning_rate), metrics=["mean_absolute_error"])
    print(f"middle_data shape: {middle_data.shape}")
    print(f"target_data shape: {target_data.shape}")
    print(f"LSTM input shape: {model.layers[0].input_dtype}")
    history = model.fit(middle_data, target_data, epochs=epochs, batch_size=batch_size, verbose=0)
    full_predictions = model.predict(middle_data).flatten()
    return model, history, full_predictions


# 计算两个模型的准确度
def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae


# 从LSTM模型预测的误差进行评估。
def Error_Evaluation(train_data, predict_train_data, n: int):
    errors = []
    for i in range(len(predict_train_data)):
        err = train_data[n + i] - predict_train_data[i]
        errors.append(err)
    return errors


# ARIMA模型参数选择以及PACF和ACF图
def Parameter_calculation(data):
    finding = auto_arima(data, trace=True)
    plot_acf(data, lags=lag)
    plt.savefig("ACF.jpg")
    plot_pacf(data, lags=lag)
    plt.savefig("PACF.jpg")
    ord = finding.order
    return ord


# ARIMA模型函数，用于预测LSTM模型的误差
def ARIMA_Model(train, len_test, ord):
    model = ARIMA(train, order=ord)
    model = model.fit()
    predictions = model.predict(start=len(train), end=len(train) + len_test, type='levels')
    full_predictions = model.predict(start=0, end=len(train) - 1, type='levels')
    return model, predictions, full_predictions


# 最终预测：LSTM预测值 + ARIMA预测的误差值
def Final_Predictions(predictions_errors, predictions):
    final_values = []
    for i in range(days):
        final_values.append(predictions_errors[i] + predictions[i])
    return final_values


# 主函数
def main():
    data = data_loader()
    plot_raw_data(data)
    train, test = data_allocation(data)
    plot_train_test(train, test)
    print(f"请输入神经网络的滞后值：{n}\n")
    # LSTM模型
    st1 = time.time()
    model, history, full_predictions = LSTM(train, n, number_nodes, learning_rate, epochs, batch_size)
    plot_predictions(train[n:], full_predictions, "LSTM预测值与实际值（训练数据集）")
    last_sequence = train[-n:].values.reshape((1, n, 1))
    predictions = []
    for i in range(days + 1):
        next_prediction = model.predict(last_sequence).flatten()[0]
        predictions.append(next_prediction)
        if i < len(test):
            actual_value = test.iloc[i]
            new_row = np.append(last_sequence[:, 1:, :], np.array([[[actual_value]]]), axis=1)
        else:
            new_row = np.append(last_sequence[:, 1:, :], np.array([[[next_prediction]]]), axis=1)
        last_sequence = new_row.reshape((1, n, 1))
    plot_predictions(test, predictions[:-1], "LSTM预测值与实际值（测试数据集）")
    errors_data = Error_Evaluation(train, full_predictions, n)
    plot_prediction_errors(errors_data)
    print(f"\n\n----------------------------- LSTM模型的{days}天预测值 -----------------------------\n\n")
    for i in range(days):
        actual_value = test.iloc[i] if i < len(test) else "没有实际值（超出范围）"
        print(f"第{i + 1}天 => 实际值：{actual_value} | 预测值：{predictions[i]}\n")
    print("\n---------------------------- LSTM模型摘要： ----------------------------\n")
    print(model.summary())
    mse, rmse, mae = calculate_accuracy(test[:days], predictions[:days])
    plot_accuracy(mse, rmse, mae)
    print("\n----------------------------- LSTM模型准确性 -----------------------------\n")
    print(f"\n均方误差（MSE）：{mse}\n均方根误差（RMSE）：{rmse}\n平均绝对误差（MAE）：{mae}\n\n")

    ord = Parameter_calculation(errors_data)
    Arima_Model, predictions_errors, full_predictions_errors = ARIMA_Model(errors_data, len(test), ord)
    print(f"\n\n---------------------------- ARIMA模型{days}天预测值 -------------------------\n\n")
    for i in range(len(predictions_errors)):
        print(f"{i + 1} : {predictions_errors[i]}\n")
    print("\n---------------------------- ARIMA模型摘要 -------------------------\n")
    print(Arima_Model.summary())
    arima_mse, arima_rmse, arima_mae = calculate_accuracy(errors_data, full_predictions_errors)
    plot_arima_accuracy(arima_mse, arima_rmse, arima_mae)

    print("\n\n--------------------------- 最终预测结果 ---------------------------------\n\n")
    final_predictions = Final_Predictions(predictions_errors, predictions)
    plot_final_predictions(test[:days], final_predictions[:days])
    for i in range(days):
        actual_value = test.iloc[i] if i < len(test) else "没有实际值（超出范围）"
        print(f"第{i + 1}天 => 实际值：{actual_value} | 预测值：{final_predictions[i]}\n")

    print("\n---------------- LSTM预测值与最终预测值的差异 ----------------\n")
    for i in range(days):
        actual_value = test.iloc[i] if i < len(test) else "没有实际值（超出范围）"
        print(
            f"\n第{i}天 => 实际值：{actual_value} | LSTM预测值：{predictions[i]} | 最终预测值（LSTM + ARIMA）：{final_predictions[i]}\n")

    print(f"\n\n---------------- 下一数据点的预测值 ------------------ \n\n")
    print(predictions[days] + predictions_errors[days])
    end1 = time.time()
    print(f"\n\n模型训练和预测所用时间：{end1 - st1:.2f}秒\n\n")

    with open("output.txt", "w+") as file:
        file.write("\n---------------- LSTM模型 ----------------\n")
        file.write(f"使用的滞后值：{lag}\n\n")
        file.write(f"训练的Epoch数：{epochs}\n\n")
        file.write(f"LSTM模型的学习率：{learning_rate}\n\n")
        file.write(f"LSTM模型的批次大小：{batch_size}\n\n")
        file.write(f"LSTM模型的节点数：{number_nodes}\n\n")
        file.write(f"神经网络的滞后值：{n}\n\n")
        file.write("\n---------------------- LSTM模型训练数据（前100个数据点）的完整预测 -------------------------\n")
        for i in range(100):
            file.write(f"{i} => 实际数据点：{train[i]} | 预测数据点：{full_predictions[i]}\n")
        file.write(f"LSTM模型摘要：\n{model.summary()}\n\n")
        file.write(f"LSTM模型历史：\n{history}\n\n")
        file.write(f"LSTM模型均方误差（MSE）：{mse}\n\n")
        file.write(f"LSTM模型均方根误差（RMSE）：{rmse}\n\n")
        file.write(f"LSTM模型平均绝对误差（MAE）：{mae}\n\n")
        file.write(f"----------------------------- LSTM模型的{days}天预测值 ----------------------------\n\n")
        for i, (actual, pred) in enumerate(zip(test[:days], predictions[:days])):
            file.write(f"第{i + 1}天 => 实际值：{actual} | 预测值：{pred}\n\n")
        file.write("\n---------------------------- ARIMA模型摘要 -------------------------\n")
        file.write(Arima_Model.summary().as_text())
        file.write(f"\n\n---------------------------- ARIMA模型{days}天预测值 -------------------------\n\n")
        for i in range(len(predictions_errors)):
            file.write(f"{i} : {predictions_errors[i]}\n")
        file.write("\n\n--------------------------- 最终预测结果 ---------------------------------\n\n")
        for i in range(days):
            actual_value = test.iloc[i] if i < len(test) else "没有实际值（超出范围）"
            file.write(f"\n第{i + 1}天 => 实际值：{actual_value} | 预测值：{final_predictions[i]}\n")
        file.write("\n---------------- LSTM预测值与最终预测值的差异 ----------------\n")
        for i in range(days):
            actual_value = test.iloc[i] if i < len(test) else "没有实际值（超出范围）"
            file.write(
                f"\n第{i}天 => 实际值：{actual_value} | LSTM预测值：{predictions[i]} | 最终预测值（LSTM + ARIMA）：{final_predictions[i]}\n")

        file.write(f"\n模型训练和预测所用时间：{end1 - st1:.2f}秒\n\n")
        file.write(f"\n\n---------------- 下一数据点的预测值 ------------------ \n\n")
        file.write(f"{predictions[days] + predictions_errors[days]}")
    print(f"输出已写入 'output.txt'")


if __name__ == '__main__':
    main()
