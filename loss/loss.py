from keras.losses import categorical_crossentropy

def k1(y_true,y_pred, soft, T):
	if soft:
		y_soft = K.softmax(y_true)/T
    return categorical_crossentropy(y_soft,y_pred)
def knowledge_distillation_loss_1(y_true,y_pred):
    y_soft_true = y_true[0]
    y_soft_pred = y_pred[0]
    return categorical_crossentropy(y_soft_true,y_soft_pred)
def knowledge_distillation_loss_2(y_true,y_pred):
    y_soft_true = y_true[1]
    y_soft_pred = y_pred[1]
    return categorical_crossentropy(y_soft_true,y_soft_pred)