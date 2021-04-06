import sys
import time

import mariadb
import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.imputation import RandomSampleImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def getAllRecordsFromDatabase(databaseName):

    connection = mariadb.connect(
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
    numpy_array = np.array(result)
    length = len(numpy_array)
    print(f'{databaseName} beolvasva, rekordok száma: {length}')
    return numpy_array[:, :]

def imputeField1Field25(numpy_array):
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


def imputeField26Ffield35(array):
    dataFrame = pd.DataFrame(array[:, 35:41])
    stringImputer = RandomSampleImputer()
    stringImputer.fit(dataFrame)
    convertedField35_Field40 = stringImputer.transform(dataFrame)
    array[:, 35:41] = convertedField35_Field40


def imputeField36Field40(numpy_array):
    dataFrame = pd.DataFrame(numpy_array[:, 47:49])
    dateImputer = RandomSampleImputer()
    dateImputer.fit(dataFrame)
    convertedField39_Field40 = dateImputer.transform(dataFrame)
    numpy_array[:, 47:49] = convertedField39_Field40

def convertTimestampToJulian(array):
    convertedTimeStampDatas = list()
    for timeStamp in array[:, 3:4]:
        t=timeStamp[0]
        ts=pd.Timestamp(t)
        convertedTimeStampToJulian=ts.to_julian_date()
        convertedTimeStampDatas.append(convertedTimeStampToJulian)
    convertedTimeStampDataArray = np.array(convertedTimeStampDatas)
    reshaped_array = convertedTimeStampDataArray.reshape(-1, 1)
    array[:, 3:4] = reshaped_array


def convertCurrencyFeature(array):
    currencies_array = array[:, 5:6]
    encoder = LabelEncoder()
    encoder.fit(currencies_array)
    encoded_currencies = encoder.transform(currencies_array)
    reshaped_encoded_currencies = encoded_currencies.reshape(-1, 1)
    array[:, 5:6] = reshaped_encoded_currencies


def convertCountryFeature(array):
    countries_array = array[:, 7:8]
    countries_encoder = LabelEncoder()
    countries_encoder.fit(countries_array)
    encoded_countries = countries_encoder.transform(countries_array)
    reshaped_encoded_countries = encoded_countries.reshape(-1, 1)
    array[:, 7:8] = reshaped_encoded_countries


def convertField26Field35(array):
    field26_array = array[:, 34:35]
    field26_encoder = LabelEncoder()
    field26_encoder.fit(field26_array)
    encoded_field26 = field26_encoder.transform(field26_array)
    reshaped_encoded_field26 = encoded_field26.reshape(-1, 1)
    array[:, 34:35] = reshaped_encoded_field26

    field27_array = array[:, 35:36]
    field27_encoder = LabelEncoder()
    field27_encoder.fit(field27_array)
    encoded_field27 = field27_encoder.transform(field27_array)
    reshaped_encoded_field27 = encoded_field27.reshape(-1, 1)
    array[:, 35:36] = reshaped_encoded_field27

    field28_array = array[:, 36:37]
    field28_encoder = LabelEncoder()
    field28_encoder.fit(field28_array)
    encoded_field28 = field28_encoder.transform(field28_array)
    reshaped_encoded_field28 = encoded_field28.reshape(-1, 1)
    array[:, 36:37] = reshaped_encoded_field28

    field29_array = array[:, 37:38]
    field29_encoder = LabelEncoder()
    field29_encoder.fit(field29_array)
    encoded_field29 = field29_encoder.transform(field29_array)
    reshaped_encoded_field29 = encoded_field29.reshape(-1, 1)
    array[:, 37:38] = reshaped_encoded_field29

    field30_array = array[:, 38:39]
    field30_encoder = LabelEncoder()
    field30_encoder.fit(field30_array)
    encoded_field30 = field30_encoder.transform(field30_array)
    reshaped_encoded_field30 = encoded_field30.reshape(-1, 1)
    array[:, 38:39] = reshaped_encoded_field30

    field31_array = array[:, 39:40]
    field31_encoder = LabelEncoder()
    field31_encoder.fit(field31_array)
    encoded_field31 = field31_encoder.transform(field31_array)
    reshaped_encoded_field31 = encoded_field31.reshape(-1, 1)
    array[:, 39:40] = reshaped_encoded_field31

    field32_array = array[:, 40:41]
    field32_encoder = LabelEncoder()
    field32_encoder.fit(field32_array)
    encoded_field32 = field32_encoder.transform(field32_array)
    reshaped_encoded_field32 = encoded_field32.reshape(-1, 1)
    array[:, 40:41] = reshaped_encoded_field32

    field33_array = array[:, 41:42]
    field33_encoder = LabelEncoder()
    field33_encoder.fit(field33_array)
    encoded_field33 = field33_encoder.transform(field33_array)
    reshaped_encoded_field33 = encoded_field33.reshape(-1, 1)
    array[:, 41:42] = reshaped_encoded_field33

    field34_array = array[:, 42:43]
    field34_encoder = LabelEncoder()
    field34_encoder.fit(field34_array)
    encoded_field34 = field34_encoder.transform(field34_array)
    reshaped_encoded_field34 = encoded_field34.reshape(-1, 1)
    array[:, 42:43] = reshaped_encoded_field34

    field35_array = array[:, 43:44]
    field35_encoder = LabelEncoder()
    field35_encoder.fit(field35_array)
    encoded_field35 = field35_encoder.transform(field35_array)
    reshaped_encoded_field35 = encoded_field35.reshape(-1, 1)
    array[:, 43:44] = reshaped_encoded_field35


def convertFromField36ToField40WithCountFrequencyEncoder(array):
    dataFrame = pd.DataFrame(array[:, 44:49])
    encoder = CountFrequencyEncoder(encoding_method='frequency')
    encoder.fit(dataFrame)
    encodedField36Field40 = encoder.transform(dataFrame)
    array[:, 44:49] = encodedField36Field40


def imputeAndConvertFeatures(array):
    convertTimestampToJulian(array)
    convertCurrencyFeature(array)
    convertCountryFeature(array)
    imputeField1Field25(array)
    imputeField26Ffield35(array)
    imputeField36Field40(array)
    convertField26Field35(array)
    convertFromField36ToField40WithCountFrequencyEncoder(array)


def createDatabase(databaseName):
    connection = mariadb.connect(
        pool_name="create_pool",
        pool_size=1,
        host="store.usr.user.hu",
        user="mki",
        password="pwd")

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
    start = time.time()
    numberOfCliParameters = len(sys.argv)
    print(f'Adatbázisok száma: {numberOfCliParameters - 1}')
    databaseNames = list()
    for i in range(1, numberOfCliParameters):
        databaseNames.append(sys.argv[i])
    for i in range(len(databaseNames)):
        print(f'Feldolgozandó adatbázis: {databaseNames[i]}')
    for databaseName in databaseNames:
        featureNumpyArray = getAllRecordsFromDatabase(databaseName)
        imputeAndConvertFeatures(featureNumpyArray)
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
