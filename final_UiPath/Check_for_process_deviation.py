import pandas as pd
from joblib import load
from io import StringIO



def seperate_activities(df_input):

    # drop columns which are not needed for the prediction
    df_input = df_input.drop(['case_ID', 'case_completed'], axis=1)

    # replace last performed activity with '-' and assign last performed activity to real_next_activity
    # start at last column and take first column from the end which has a string as entry, assign to real_next_activity
    for j in range(len(df_input.columns) - 1):
        if df_input.iloc[0, len(df_input.columns) - 1 - j] != '-':
            print("Old log entry: " + df_input.iloc[0, len(df_input.columns) - 1 - j])
            real_next_activity = df_input.iloc[0, len(df_input.columns) - 1 - j]
            df_input.iloc[0, len(df_input.columns)-1-j] = '-'
            break

    print("New log entry: " + df_input.iloc[0, len(df_input.columns)-1-j])
    print("Real next activity: " + real_next_activity)
    print("Dataframe after separating")
    print(df_input)

    return df_input, real_next_activity


def modify_input(df_input, df_unique_values):

    list_ml_input = []

    # add scan quality to list for ML input
    list_ml_input.append(df_input.at[0, 'scan_quality'])

    # add networth to list for ML input
    list_ml_input.append(df_input.at[0, 'net_worth'])

    # use unique activities of each column to encode categorical values of bot input
    # for encoding the categorical input for the ML-model: drop columns 'unnamed' and 'output', 'scan_quality' and 'net_worth'
    df_unique_values_input = df_unique_values.drop(['Unnamed: 0', 'output'], axis=1)
    df_categorical_input = df_input.drop(['scan_quality','net_worth'], axis=1)

    # encoding the string input for the identification of process deviations according to the input encoding used for the ML-model creation
    for col in df_unique_values_input:

        # first check if values of the bot input are valid based on their position
        if df_categorical_input[col][0] in df_unique_values_input[col].values: print('Element exists in Dataframe')
        else:
            print('Element exists not in Dataframe')
            return None, "Prediction is not possible"
            break

        # encode categorical bot input to ML input
        for i in range(df_unique_values_input[col].count()):
            if df_unique_values_input[col][i] == df_categorical_input[col][0]:
                list_ml_input.append(1.0)
            else:
                list_ml_input.append(0.0)

    # if prediction is possible return df and "prediciton possible". If no prediction is possible, one of the above return statements is used
    return list_ml_input, "Prediction possible"


def predict_next_activity(list_ml_input, trained_classifier, df_unique_values):

    # predict next activities with trained classifier (using the classifier_decision tree-model as trained_classifier)
    pred_next_activity = trained_classifier.predict([list_ml_input])
    print("Result of the prediction:")
    print(pred_next_activity)
    pred_prob_next_activity = trained_classifier.predict_proba([list_ml_input])
    print("Result probabilities of the prediction:")
    print(pred_prob_next_activity)

    # create a list of potential next activities by encoding the prediction result; the probability must be above the threshold to be added to the list
    list_potential_next_activities = []

    # threshold for classification as a process deviation
    threshold = 0.1

    # create from prediction result a list of potential next activities
    df_unique_values_output = df_unique_values['output']
    for i in range (len(pred_prob_next_activity)):
        if pred_prob_next_activity[i][0][1] > threshold: list_potential_next_activities.append(df_unique_values_output[i])

    return list_potential_next_activities


def check_on_process_devition(list_potential_next_activities, real_next_activity):

    print("List of potential next activities: ")
    print(list_potential_next_activities)
    print("Real next activity: " + real_next_activity)

    # false = no deviation; true = deviation occurs
    if real_next_activity in list_potential_next_activities: return False
    else: return True


def coordination(string_bot_input):

    df_bot_input = pd.read_csv(StringIO(string_bot_input), header=0)
    df_unique_values = pd.read_csv("C:\\Users\\carin\\OneDrive\\Dokumente\\final_UiPath\\df_unique_values.csv")
    trained_classifier = load("C:\\Users\\carin\\OneDrive\\Dokumente\\final_UiPath\\classifier_decision_tree-model.joblib")

    df_input, real_next_activity = seperate_activities(df_bot_input)
    list_ml_input, prediction_possible = modify_input(df_input, df_unique_values)
    if prediction_possible == "Prediction is not possible":
        print("Prediction is not possible")
        return True
    else:
        list_potential_next_activities = predict_next_activity(list_ml_input, trained_classifier, df_unique_values)
        boolean = check_on_process_devition(list_potential_next_activities, real_next_activity)
        print("Does a process deviation exist?")
        return boolean

# delete hash from the following statement to see how the code works
#print(coordination("case_completed,case_ID,scan_quality,net_worth,document_type,item_type,input0,input1,input2,input3,input4,input5,input6\nNo,4507001213_00010,2,18290.0,Standard PO,Service,Create Purchase Requisition Item,Create Purchase Order Item,Record Goods Receipt,Vendor creates invoice,-,-,-"))