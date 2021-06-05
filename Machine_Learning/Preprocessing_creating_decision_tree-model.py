import pandas as pd
import random

import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.timestamp import timestamp_filter

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump

import matplotlib.pyplot as plt
from sklearn import tree



def readLogAndFilter(input_log_path):
    print("Start reading log")

    # reading log
    log = pm4py.read_xes(input_log_path)

    # filter log according to timestamp and variants
    print("Filter event log")
    log2 = timestamp_filter.filter_traces_contained(log, "2018-01-01 00:00:00", "2020-12-31 23:59:59")
    filtered_log = variants_filter.filter_log_variants_percentage(log2, percentage=0.7)

    # converting log into dataframe
    df_log = log_converter.apply(filtered_log, variant=log_converter.Variants.TO_DATA_FRAME)
    print("Event log is converted into dataframe")

    # change the csv dataframe to a smaller dataframe (with only few columns) with understandable column names
    df_log = df_log[['case:concept:name', 'concept:name', 'Cumulative net worth (EUR)', 'time:timestamp', 'case:Document Type', 'case:Item Type']]
    df_log = df_log.rename(columns={'case:concept:name': 'case_ID', 'Cumulative net worth (EUR)': 'net_worth', 'concept:name': 'activity_name', 'time:timestamp':'timestamp', 'case:Document Type': 'document_type', 'case:Item Type': 'item_type'})
    print("Part-log is created")

    # save part-log to csv
    df_log.to_csv('part-log.csv')

    return df_log


def modify_dataframe(df):
    print("Start transforming structure of the part-log")

    list_preprocessed = []

    # group dataframe created from log by Case_ID
    grouped_dataframes = df.groupby('case_ID', sort=False)

    for case_ID, group in grouped_dataframes:

        # sort dataframe according to timestamp
        sorted_group = group.sort_values(by=['timestamp'])

        # reset index
        sorted_group.index = sorted_group.index - sorted_group.index[0]

        # create random value for scan quality
        scan_rand = random.randint(1, 3)

        # define input activities / input / x and build a list
        for event in range(len(sorted_group)):

            # write column input 0
            if event == 0:
                if len(sorted_group) == 1:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[0, 'activity_name'],
                        'input1': "-",
                        'input2': "-",
                        'input3': "-",
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': "Process End"})

                if len(sorted_group) > 1:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[0, 'activity_name'],
                        'input1': "-",
                        'input2': "-",
                        'input3': "-",
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': sorted_group.at[event + 1, 'activity_name']})

            # write column input 0, input 1
            elif event == 1:
                # long explanation: when only two activities exist (len = 2), write the second activity in a new column, the output is 'Process End' (no next activity, because after the second activity the process ends)
                if len(sorted_group) == 2:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 1, 'activity_name'],
                        'input1': sorted_group.at[event, 'activity_name'],
                        'input2': "-",
                        'input3': "-",
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': "Process End"})

                # long explanation: when more than two activites exist, write the second activity in a new column, the output is the next activity
                if len(sorted_group) > 2:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 1, 'activity_name'],
                        'input1': sorted_group.at[event, 'activity_name'],
                        'input2': "-",
                        'input3': "-",
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': sorted_group.at[event + 1, 'activity_name']})

            # write column input 0, input 1, input 2
            elif event == 2:
                if len(sorted_group) == 3:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 2, 'activity_name'],
                        'input1': sorted_group.at[event - 1, 'activity_name'],
                        'input2': sorted_group.at[event, 'activity_name'],
                        'input3': "-",
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': "Process End"})

                if len(sorted_group) > 3:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 2, 'activity_name'],
                        'input1': sorted_group.at[event - 1, 'activity_name'],
                        'input2': sorted_group.at[event, 'activity_name'],
                        'input3': "-",
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': sorted_group.at[event + 1, 'activity_name']})

            # write column input 0, input 1, input 2, input 3
            elif event == 3:
                if len(sorted_group) == 4:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 3, 'activity_name'],
                        'input1': sorted_group.at[event - 2, 'activity_name'],
                        'input2': sorted_group.at[event - 1, 'activity_name'],
                        'input3': sorted_group.at[event, 'activity_name'],
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': "Process End"})

                if len(sorted_group) > 4:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 3, 'activity_name'],
                        'input1': sorted_group.at[event - 2, 'activity_name'],
                        'input2': sorted_group.at[event - 1, 'activity_name'],
                        'input3': sorted_group.at[event, 'activity_name'],
                        'input4': "-",
                        'input5': "-",
                        'input6': "-",
                        'output': sorted_group.at[event + 1, 'activity_name']})

            # write column input 0, input 1, input 2, input 3, input 4
            elif event == 4:
                if len(sorted_group) == 5:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 4, 'activity_name'],
                        'input1': sorted_group.at[event - 3, 'activity_name'],
                        'input2': sorted_group.at[event - 2, 'activity_name'],
                        'input3': sorted_group.at[event - 1, 'activity_name'],
                        'input4': sorted_group.at[event, 'activity_name'],
                        'input5': "-",
                        'input6': "-",
                        'output': "Process End"})

                if len(sorted_group) > 5:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 4, 'activity_name'],
                        'input1': sorted_group.at[event - 3, 'activity_name'],
                        'input2': sorted_group.at[event - 2, 'activity_name'],
                        'input3': sorted_group.at[event - 1, 'activity_name'],
                        'input4': sorted_group.at[event, 'activity_name'],
                        'input5': "-",
                        'input6': "-",
                        'output': sorted_group.at[event + 1, 'activity_name']})

            # write column input 0, input 1, input 2, input 3, input 4, input 5
            elif event == 5:
                if len(sorted_group) == 6:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 5, 'activity_name'],
                        'input1': sorted_group.at[event - 4, 'activity_name'],
                        'input2': sorted_group.at[event - 3, 'activity_name'],
                        'input3': sorted_group.at[event - 2, 'activity_name'],
                        'input4': sorted_group.at[event - 1, 'activity_name'],
                        'input5': sorted_group.at[event, 'activity_name'],
                        'input6': "-",
                        'output': "Process End"})

                if len(sorted_group) > 6:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 5, 'activity_name'],
                        'input1': sorted_group.at[event - 4, 'activity_name'],
                        'input2': sorted_group.at[event - 3, 'activity_name'],
                        'input3': sorted_group.at[event - 2, 'activity_name'],
                        'input4': sorted_group.at[event - 1, 'activity_name'],
                        'input5': sorted_group.at[event, 'activity_name'],
                        'input6': "-",
                        'output': sorted_group.at[event + 1, 'activity_name']})

            # write column input 0, input 1, input 2, input 3, input 4, input 5, input 6
            elif event == 6:
                if len(sorted_group) == 7:
                    list_preprocessed.append({
                        'case_ID': case_ID,
                        'scan_quality': scan_rand,
                        'net_worth': sorted_group.at[0, 'net_worth'],
                        'document_type': sorted_group.at[0, 'document_type'],
                        'item_type': sorted_group.at[0, 'item_type'],
                        'input0': sorted_group.at[event - 6, 'activity_name'],
                        'input1': sorted_group.at[event - 5, 'activity_name'],
                        'input2': sorted_group.at[event - 4, 'activity_name'],
                        'input3': sorted_group.at[event - 3, 'activity_name'],
                        'input4': sorted_group.at[event - 2, 'activity_name'],
                        'input5': sorted_group.at[event - 1, 'activity_name'],
                        'input6': sorted_group.at[event, 'activity_name'],
                        'output': "Process End"})

                if len(sorted_group) > 7:
                    print("sorted group > 7")
                    print(case_ID)

            else:
                print("Input range too long")

    # save list as dataframe
    df_preprocessed = pd.DataFrame(list_preprocessed,
                                   columns=["case_ID", "scan_quality", "net_worth", "document_type",
                                            "item_type", "input0", "input1", "input2", "input3", "input4", "input5", "input6",
                                            "output"])

    # save df_preprocessed as csv (training data before encoding)
    df_preprocessed.to_csv('training_data_before_encoding.csv')

    # create dataframe of unique values in each column
    df_to_extract_unique_values = df_preprocessed.drop(['case_ID', 'scan_quality', 'net_worth'], axis=1)
    df_unique_values = df_to_extract_unique_values.apply(lambda x: pd.Series(pd.unique(x)))

    # sort unique values in each column by alphabet
    print("Create df_unique_values")
    for col in df_unique_values:
        df_unique_values[col] = df_unique_values[col].sort_values(ignore_index=True)

    # save dataframe of unique values as csv
    df_unique_values.to_csv('df_unique_values.csv')

    return df_preprocessed


def encoding_categorical_values(df_preprocessed):
    print("Encoding input and output")

    # drop columns which are not needed for the prediction afterwards
    df_preprocessed = df_preprocessed.drop(['case_ID'], axis=1)

    # drop columns output and which are numerical - only categorical columns are encoded
    df_input_categorical = df_preprocessed.drop(['output', 'scan_quality', 'net_worth'], axis=1)

    # encode categorical columns of the input with One Hot Encoding
    encoder = OneHotEncoder()
    df_input_enc = pd.DataFrame(encoder.fit_transform(df_input_categorical).todense(),
                                columns=encoder.get_feature_names())

    # for encoding drop all columns except for output
    df_output_categorical = df_preprocessed['output']

    # encode categorical output columns with LabelBinarizer
    enc = LabelBinarizer()
    df_output_enc = pd.DataFrame(enc.fit_transform(df_output_categorical))
    df_output_enc.columns = enc.classes_

    # combine from input and output dataframe the encoded categorical columns with the numerical columns
    df_for_decision_tree_model = pd.concat([df_preprocessed, df_input_enc, df_output_enc], axis=1)

    # delete original categorical columns because they are encoded as new columns
    df_for_decision_tree_model = df_for_decision_tree_model.drop(
        ['output', 'input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'document_type',
         'item_type'], axis=1)

    # delete the rows where the networth contains true or false
    df_for_decision_tree_model = df_for_decision_tree_model[df_for_decision_tree_model.net_worth != 'True']
    df_for_decision_tree_model = df_for_decision_tree_model[df_for_decision_tree_model.net_worth != 'False']

    return enc, df_for_decision_tree_model


def building_decision_tree_model(enc, df_for_decision_tree_model):
    print("Separating input and output; splitting into training and test set")

    # save training_data as csv
    df_for_decision_tree_model.to_csv('training_data_for_decision_tree-model.csv')

    # generate input(X) and output(y) data frames
    y_feature_names = enc.classes_

    X = df_for_decision_tree_model.drop(columns=y_feature_names)
    print("Input columns")
    print(X.columns.ravel())
    X.to_csv('X_dataset')

    y = df_for_decision_tree_model[y_feature_names]
    print("Output columns")
    print(y.columns.ravel())

    # split dataset in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Apply grid search and train classifier")

    # grid search setting parameters
    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [8, 9, 10],
        "max_leaf_nodes": [32, 33, 34],
        "min_samples_leaf": [1, 2, 3],
        "min_samples_split": [2, 3, 4]
    }

    # train the classifier with best paramater of grid search
    clf = DecisionTreeClassifier()
    grid_search_clf = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search_clf.fit(X_train, y_train)
    print("Best estimator")
    print(grid_search_clf.best_estimator_)
    print("Best parameters")
    print(grid_search_clf.best_params_)
    print("Best score")
    print(grid_search_clf.best_score_)

    # save trained classifier
    dump(grid_search_clf, 'classifier_decision_tree-model.joblib')

    # test prediction and predict output of X_test
    print("Test the model on dataset X_test")
    y_test_pred = grid_search_clf.predict(X_test)

    # calculate accuracy of the model
    print('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, y_test_pred)))

    return X_train, y_train, grid_search_clf


def plot_decision_tree_model(grid_search_clf, X_train, y_train):
    print("Plotting decision tree-model")

    fig = plt.figure(figsize=(75, 70), dpi = 300)
    _ = tree.plot_tree(grid_search_clf.best_estimator_,
                       feature_names=X_train.columns,
                       class_names=y_train.columns,
                       filled=True)

    fig.savefig("decision_tree-model.jpeg")


df_log = readLogAndFilter('BPI_Challenge_2019.xes')
df_preprocessed = modify_dataframe(df_log)
enc, df_for_decision_tree_model = encoding_categorical_values(df_preprocessed)
X_train, y_train, grid_search_clf = building_decision_tree_model(enc, df_for_decision_tree_model)
plot_decision_tree_model(grid_search_clf, X_train, y_train)