import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    # Read data in from file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidence = []
        labels = []
        for row in reader:
            evidence.append([getCellValue(cell) for cell in row[:17]])
            labels.append(0 if row[17] == "FALSE" else 1)

    return (evidence,labels)
    # raise NotImplementedError

def getCellValue(cell):

    if cell == "FALSE" or cell == "New_Visitor" or cell == "Jan" or cell == "Other":
        return 0
    elif cell == "TRUE" or cell == "Returning_Visitor" or cell == "Feb":
        return 1
    elif cell == "Mar":
        return 2
    elif cell == "Apr":
        return 3
    elif cell == "May":
        return 4
    elif cell == "June":
        return 5
    elif cell == "Jul":
        return 6
    elif cell == "Aug":
        return 7
    elif cell == "Sep":
        return 8
    elif cell == "Oct":
        return 9
    elif cell == "Nov":
        return 10
    elif cell == "Dec":
        return 11
    else:
        return float(cell)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors=1)   
    training = model.fit(evidence,labels)
    return training
    # raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    c_sens = 0
    t_sens = 0

    c_spec = 0
    t_spec = 0


    no_of_pred = len(labels)

    for i in range(no_of_pred):
        if labels[i] == 1:
            t_sens +=1
            if labels[i] == predictions[i]:
                c_sens += 1
        else:
            t_spec += 1
            if labels[i] == predictions[i]:
                c_spec +=1

    sensitivity = c_sens/t_sens
    specificity = c_spec/t_spec

    return (sensitivity, specificity)

    # raise NotImplementedError


if __name__ == "__main__":
    main()
