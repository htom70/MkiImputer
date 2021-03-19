# coding=<utf-8>

import os

import mariadb
import pandas as pd
import time
from time import strptime, strftime, mktime, gmtime
from multiprocessing import Process, Pipe, set_start_method, Pool, Manager
from datetime import datetime
import concurrent.futures
import multiprocessing
from multiprocessing.spawn import freeze_support
from multiprocessing import Process
import mysql.connector
import numpy as np
from imblearn.combine import SMOTEENN
from jdcal import gcal2jd
from numpy import random
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import RandomSampleImputer
from sklearn.preprocessing import LabelEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d

# from user.helper import create_pipelines, run_cv_and_test, run_cv_and_test_hypertuned_params
# from user.helper import get_hypertune_params
# from user.helper import create_pipelines, run_cv_and_test, get_hypertune_params, run_cv_and_test_hypertuned_params

seed = 1234
num_folds = 10
n_jobs = -1
hypertuned_experiment = False
is_save_results = True


def getAllRecordsFromDatabase(databaseName):
    # connection = mysql.connector.connect(
    #     host="localhost",
    #     user="root",
    #     password="TOmi_1970",
    #     database="retired_transaction")

    connection = mariadb.connect(
        # pool_name="read_pull",
        # pool_size=1,
        host="store.usr.user.hu",
        user="mki",
        password="pwd",
        database=databaseName
    )
    print(connection)
    sql_select_Query = "select * from transaction order by timestamp"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    result = cursor.fetchall()
    # connection.close()
    numpy_array = np.array(result)
    length = len(numpy_array)
    print(f'{databaseName} beolvasva, rekordok száma: {length}')
    return numpy_array[:, :]


def impute_not_number_array(field_array):
    lentgh = len(field_array)
    index_range = 5
    for i in range(lentgh):
        field_value = field_array[i]
        while field_value[0] is None:
            random_index = random.randint(index_range)
            test_plus_index = i + random_index
            if test_plus_index < lentgh:
                test_array_value = field_array[test_plus_index]
                if test_array_value[0] is not None:
                    field_array[i] = test_array_value
                else:
                    random_index = random.randint(index_range)
                    test_minus_index = i - random_index
                    if test_minus_index >= 0:
                        test_array_value = field_array[test_minus_index]
                        if test_array_value[0] is not None:
                            field_array[i] = test_array_value


def impute_field1_field25(numpy_array):
    # print("Impute field1 - field25.")
    field3_field6 = numpy_array[:, 11:15]
    number_imputer_field3_6 = SimpleImputer(strategy='mean')
    number_imputer_field3_6.fit(field3_field6)
    convertedField3_Field6 = number_imputer_field3_6.transform(field3_field6)
    numpy_array[:, 11:15] = convertedField3_Field6
    field9_field12 = numpy_array[:, 17:21]
    number_imputer_field9_12 = SimpleImputer(strategy='mean')
    number_imputer_field9_12.fit(field9_field12)
    convertedField9_Field12 = number_imputer_field9_12.transform(field9_field12)
    numpy_array[:, 17:21] = convertedField9_Field12
    field14_field20 = numpy_array[:, 22:29]
    number_imputer_field14_20 = SimpleImputer(strategy='mean')
    number_imputer_field14_20.fit(field14_field20)
    convertedField14_Field20 = number_imputer_field14_20.transform(field14_field20)
    numpy_array[:, 22:29] = convertedField14_Field20
    field22_field25 = numpy_array[:, 30:34]
    number_imputer_field22_25 = SimpleImputer(strategy='mean')
    number_imputer_field22_25.fit(field22_field25)
    convertedField22_Field25 = number_imputer_field22_25.transform(field22_field25)
    numpy_array[:, 30:34] = convertedField22_Field25


def impute_field26_field35(numpy_array):
    # print("Impute field26 - field35.")
    dataFrame = pd.DataFrame(numpy_array[:, 35:41])
    stringImputer = RandomSampleImputer()
    stringImputer.fit(dataFrame)
    convertedField35_Field40 = stringImputer.transform(dataFrame)
    numpy_array[:, 35:41] = convertedField35_Field40


def impute_field36_field40(numpy_array):
    # print("Impute field39 - field40.")
    dataFrame = pd.DataFrame(numpy_array[:, 47:49])
    dateImputer = RandomSampleImputer()
    dateImputer.fit(dataFrame)
    convertedField39_Field40 = dateImputer.transform(dataFrame)
    numpy_array[:, 47:49] = convertedField39_Field40


def convert_timestamp_to_epoch(timestamp_array):
    result_array = np.array([])
    for timestamp in timestamp_array:
        epoch_time_in_seconds = (timestamp[0] - datetime(1970, 1, 1, )).total_seconds()
        result_array = np.append(result_array, epoch_time_in_seconds)
    return result_array


def convertTimeStampToJulian(numpyArray):
    convertedTimeStampDatas = list()
    for timeStamp in numpyArray[:, 3:4]:
        convertedTimeStampToJulian=gcal2jd(timeStamp)
        t=timeStamp[0]
        ts=pd.Timestamp(t)
        convertedTimeStampToJulian=ts.to_julian_date()
        convertedTimeStampDatas.append(convertedTimeStampToJulian)
    convertedTimeStampDataArray = np.array(convertedTimeStampDatas)
    reshaped_array = convertedTimeStampDataArray.reshape(-1, 1)
    numpyArray[:, 3:4] = reshaped_array


def convert_timestamp_feature(numpy_array):
    # print("Convert timestamp to epoch time.")
    timestamp_features = numpy_array[:, 3:4]
    epoch_array = convert_timestamp_to_epoch(timestamp_features)
    reshaped_array = epoch_array.reshape(-1, 1)
    numpy_array[:, 3:4] = reshaped_array


def convert_currency_feature(numpy_array):
    # print("Convert currency name to number.")
    currencies_array = numpy_array[:, 5:6]
    encoder = LabelEncoder()
    encoder.fit(currencies_array)
    encoded_currencies = encoder.transform(currencies_array)
    reshaped_encoded_currencies = encoded_currencies.reshape(-1, 1)
    numpy_array[:, 5:6] = reshaped_encoded_currencies


def convert_country_feature(numpy_array):
    # print("Convert country name to number.")
    countries_array = numpy_array[:, 7:8]
    countries_encoder = LabelEncoder()
    countries_encoder.fit(countries_array)
    encoded_countries = countries_encoder.transform(countries_array)
    reshaped_encoded_countries = encoded_countries.reshape(-1, 1)
    numpy_array[:, 7:8] = reshaped_encoded_countries


def convert_field26_field35(numpy_array):
    # print("Convert field26 - field35 strings to number.")
    field26_array = numpy_array[:, 34:35]
    field26_encoder = LabelEncoder()
    field26_encoder.fit(field26_array)
    encoded_field26 = field26_encoder.transform(field26_array)
    reshaped_encoded_field26 = encoded_field26.reshape(-1, 1)
    numpy_array[:, 34:35] = reshaped_encoded_field26

    field27_array = numpy_array[:, 35:36]
    field27_encoder = LabelEncoder()
    field27_encoder.fit(field27_array)
    encoded_field27 = field27_encoder.transform(field27_array)
    reshaped_encoded_field27 = encoded_field27.reshape(-1, 1)
    numpy_array[:, 35:36] = reshaped_encoded_field27

    field28_array = numpy_array[:, 36:37]
    field28_encoder = LabelEncoder()
    field28_encoder.fit(field28_array)
    encoded_field28 = field28_encoder.transform(field28_array)
    reshaped_encoded_field28 = encoded_field28.reshape(-1, 1)
    numpy_array[:, 36:37] = reshaped_encoded_field28

    field29_array = numpy_array[:, 37:38]
    field29_encoder = LabelEncoder()
    field29_encoder.fit(field29_array)
    encoded_field29 = field29_encoder.transform(field29_array)
    reshaped_encoded_field29 = encoded_field29.reshape(-1, 1)
    numpy_array[:, 37:38] = reshaped_encoded_field29

    field30_array = numpy_array[:, 38:39]
    field30_encoder = LabelEncoder()
    field30_encoder.fit(field30_array)
    encoded_field30 = field30_encoder.transform(field30_array)
    reshaped_encoded_field30 = encoded_field30.reshape(-1, 1)
    numpy_array[:, 38:39] = reshaped_encoded_field30

    field31_array = numpy_array[:, 39:40]
    field31_encoder = LabelEncoder()
    field31_encoder.fit(field31_array)
    encoded_field31 = field31_encoder.transform(field31_array)
    reshaped_encoded_field31 = encoded_field31.reshape(-1, 1)
    numpy_array[:, 39:40] = reshaped_encoded_field31

    field32_array = numpy_array[:, 40:41]
    field32_encoder = LabelEncoder()
    field32_encoder.fit(field32_array)
    encoded_field32 = field32_encoder.transform(field32_array)
    reshaped_encoded_field32 = encoded_field32.reshape(-1, 1)
    numpy_array[:, 40:41] = reshaped_encoded_field32

    field33_array = numpy_array[:, 41:42]
    field33_encoder = LabelEncoder()
    field33_encoder.fit(field33_array)
    encoded_field33 = field33_encoder.transform(field33_array)
    reshaped_encoded_field33 = encoded_field33.reshape(-1, 1)
    numpy_array[:, 41:42] = reshaped_encoded_field33

    field34_array = numpy_array[:, 42:43]
    field34_encoder = LabelEncoder()
    field34_encoder.fit(field34_array)
    encoded_field34 = field34_encoder.transform(field34_array)
    reshaped_encoded_field34 = encoded_field34.reshape(-1, 1)
    numpy_array[:, 42:43] = reshaped_encoded_field34

    field35_array = numpy_array[:, 43:44]
    field35_encoder = LabelEncoder()
    field35_encoder.fit(field35_array)
    encoded_field35 = field35_encoder.transform(field35_array)
    reshaped_encoded_field35 = encoded_field35.reshape(-1, 1)
    numpy_array[:, 43:44] = reshaped_encoded_field35


def convert_date_to_julian_date(current_date):
    year = current_date.strftime("%G")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")
    julian_date = sum(gcal2jd(year, month, day))
    return julian_date


def convert_date_array_to_julian_date_array(field_array):
    result_array = np.array([])
    for field in field_array:
        julian_date = convert_date_to_julian_date(field[0])
        result_array = np.append(result_array, julian_date)
    return result_array


def convertDateToJulianDateSimpleThread(f):
    fields = f
    print(f"fields in converter: {fields}")
    convertedField = convert_date_array_to_julian_date_array(fields)
    reshapedEncodedField = convertedField.reshape(-1, 1)
    print(f"reshaped converted fields in converter: {reshapedEncodedField}")
    f = reshapedEncodedField


def modifiedConvertDateToJulianDateSimpleThread(inList, outList):
    # print(f"fields in converter: {inList}")
    convertedField = convert_date_array_to_julian_date_array(inList)
    outList.extend(convertedField)
    # print(f"converted fields in converter: {outList}")


def convertFromField36ToField40WithCountFrequencyEncoder(featureNumpyArray):
    dataFrame = pd.DataFrame(featureNumpyArray[:, 44:49])
    encoder = CountFrequencyEncoder(encoding_method='frequency')
    encoder.fit(dataFrame)
    encodedField36Field40 = encoder.transform(dataFrame)
    featureNumpyArray[:, 44:49] = encodedField36Field40


def parallelConvertFromField36ToField40(featureNumpyArray):
    for fieldIndex in range(44, 49, 1):
        currentFields = featureNumpyArray[:, fieldIndex:fieldIndex + 1]
        length = len(currentFields)
        indexBound = int(length / cpuCount)
        start = time.time()
        with Manager() as manager:
            resultList = list()
            inFieldDictionary = dict()
            outFieldDictionary = dict()
            processDictionary = dict()
            for i in range(cpuCount):
                # inputList = manager.list()
                outputList = manager.list()
                fieldItems = manager.list()
                if i < cpuCount - 1:
                    fieldItems.extend(currentFields[i * indexBound:(i + 1) * indexBound, :])
                else:
                    fieldItems.extend(currentFields[i * indexBound:, :])
                print(f"{i}-edik lista hossza {len(fieldItems)}")
                inFieldDictionary[i] = fieldItems
                outFieldDictionary[i] = outputList
                processDictionary[i] = Process(target=modifiedConvertDateToJulianDateSimpleThread,
                                               args=(inFieldDictionary[i], outFieldDictionary[i],))
            print(f"Processzek száma: {len(processDictionary)}")
            for i in range(cpuCount):
                processDictionary.get(i).start()
            for i in range(cpuCount):
                processDictionary.get(i).join()
            for i in range(cpuCount):
                resultList.extend(outFieldDictionary.get(i))
            # print(f"Listák összesítve: {resultList}")
            t = np.array(resultList)
            reshapedConvertedFields = t.reshape(-1, 1)
            featureNumpyArray[:, fieldIndex:fieldIndex + 1] = reshapedConvertedFields
            print("párhuzamos feldolgozás vége")
        end = time.time()
        elapsedTime = end - start
        print(f'Többszálú konverziós idő: {elapsedTime}')


def impute_and_convert_features(array):
    # print("Impute and Conversion begin")
    start = time.time()
    # convert_timestamp_feature(array)
    convertTimeStampToJulian(array)
    convert_currency_feature(array)
    convert_country_feature(array)
    impute_field1_field25(array)
    impute_field26_field35(array)
    impute_field36_field40(array)
    convert_field26_field35(array)
    end = time.time()
    # print("Impute and Conversion end")
    elapsed_time = end - start
    # print(f"Impute and Conversion time: {elapsed_time}")


def createDatabase(databaseName):
    connection = mariadb.connect(
        pool_name="create_pool",
        pool_size=1,
        host="store.usr.user.hu",
        user="mki",
        password="pwd")

    # connection = mysql.connector.connect(
    #     pool_name="create_pool",
    #     pool_size=1,
    #     host="localhost",
    #     user="root",
    #     password="TOmi_1970")

    cursor = connection.cursor()
    sqlDropDatabaseScript = "DROP DATABASE IF EXISTS " + databaseName
    cursor.execute(sqlDropDatabaseScript)
    connection.commit()
    sqlCreateSchemaScript = "CREATE DATABASE IF NOT EXISTS " + databaseName + " CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    cursor.execute(sqlCreateSchemaScript)
    connection.commit()
    cursor.execute("USE " + databaseName)
    file = open("SQL create table transaction.txt", "r")
    sqlCreataTableScript = file.read()
    cursor.execute("DROP TABLE IF EXISTS transaction")
    cursor.execute(sqlCreataTableScript)
    connection.commit()
    connection.close()
    print(f"{databaseName} created")


if __name__ == '__main__':
    # freeze_support(
    start = time.time()
    # set_start_method('spawn')
    cpuCount = multiprocessing.cpu_count()
    print(multiprocessing.cpu_count())
    # databaseNames = ["card_10000_5"]
    databaseNames = ["card_10000_5", "card_100000_1", "card_250000_02", "card"]
    # databaseNames = ["card_100000_1"]
    # databaseNames = ["card_100000_1", "card_250000_02", "card"]
    # databaseNames = ["card"]
    for databaseName in databaseNames:
        featureNumpyArray = getAllRecordsFromDatabase(databaseName)
        impute_and_convert_features(featureNumpyArray)
        convertFromField36ToField40WithCountFrequencyEncoder(featureNumpyArray)
        feature_array = featureNumpyArray[:, 1:49]
        binary_array = featureNumpyArray[:, 49:]
        valuesArray = featureNumpyArray[:, 1:]
        imputedDatebaseName = databaseName + "_i"
        createDatabase(imputedDatebaseName)

        imputedConnection = mariadb.connect(
        pool_size=32,
        host="store.usr.user.hu",
        user="mki",
        password="pwd",
        database=imputedDatebaseName)
        sql_insert_Query = "INSERt INTO transaction (card_number,transaction_type,timestamp,amount,currency_name,response_code,country_name,vendor_code," \
                           "field1,field2,field3,field4,field5,field6,field7,field8,field9,field10,field11,field12,field13,field14,field15,field16,field17," \
                           "field18,field19,field20,field21,field22,field23,field24,field25,field26,field27,field28,field29,field30,field31,field32,field33," \
                           "field34,field35,field36,field37,field38,field39,field40,fraud) VALUES " \
                           "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
                           "%s,%s,%s,%s,%s,%s,%s,%s)"
        cursor = imputedConnection.cursor()
        length = len(valuesArray)
        print(f'{imputedDatebaseName} adatbázis rekordok száma: {length}')
        bound = 1000
        if length > bound:
            numberOfPartArray = int(length / bound)
            numberOfRestDatas = length - numberOfPartArray * bound
            for i in range(0, numberOfPartArray, 1):
                tempArray = valuesArray[i * bound:(i + 1) * bound, :]
                valueList = list()
                for record in tempArray:
                    valueList.append(tuple(record))
                cursor.executemany(sql_insert_Query, valueList)
                imputedConnection.commit()
            tempArray = valuesArray[(numberOfPartArray) * bound:(numberOfPartArray) * bound + numberOfRestDatas, :]
            valueList = list()
            for record in tempArray:
                valueList.append(tuple(record))
            cursor.executemany(sql_insert_Query, valueList)
            imputedConnection.commit()
        else:
            valueList = list()
            for record in valuesArray:
                valueList.append(tuple(record))
            cursor.executemany(sql_insert_Query, valueList)
            imputedConnection.commit()
        imputedConnection.close()



    endOfConversionAndSaveToDatabase = time.time()
    elapsedTime = endOfConversionAndSaveToDatabase - start
    print(f"Teljes feldolgozási idő beolvasás, konvertálás, mentés adatbázisba: {elapsedTime}")
