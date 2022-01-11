from keras.metrics import categorical_accuracy

def knowledge_distillation_acc_1(y_true,y_pred):
    y_soft_true = y_true[0]
    y_soft_pred = y_pred[0]
    return categorical_accuracy(y_soft_true,y_soft_pred)
def knowledge_distillation_acc_2(y_true,y_pred):
    y_soft_true = y_true[1]
    y_soft_pred = y_pred[1]
    return categorical_accuracy(y_soft_true,y_soft_pred)