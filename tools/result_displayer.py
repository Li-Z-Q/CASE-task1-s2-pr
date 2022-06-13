from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

def display_result(gold_labels, pre_labels):
    print("accuracy_score:  ", accuracy_score(gold_labels, pre_labels))
    # print("matthews_corrcoef", matthews_corrcoef(gold_labels, pre_labels))
    print("precision_score: ", precision_score(gold_labels, pre_labels))
    print("recall_score:    ", recall_score(gold_labels, pre_labels))
    print("f1_score:        ", f1_score(gold_labels, pre_labels))
    print("confusion_matrix:\n", confusion_matrix(gold_labels, pre_labels))

    result = dict()
    result['a'] = accuracy_score(gold_labels, pre_labels)
    result['f'] = f1_score(gold_labels, pre_labels)
    result['r'] = recall_score(gold_labels, pre_labels)
    result['p'] = precision_score(gold_labels, pre_labels)
    # result['m'] = matthews_corrcoef(gold_labels, pre_labels)
    result['c'] = confusion_matrix(gold_labels, pre_labels)

    return result