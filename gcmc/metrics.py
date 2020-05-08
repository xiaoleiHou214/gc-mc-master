import tensorflow as tf
import numpy as np


def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(labels, np.int64))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    # if class_values is None:
    #     scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
    #     y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    # else:
    #     scores = class_values
    #     y = tf.gather(class_values, labels)

    print('#' * 50)
    print('logits.shape:', logits.shape)
   # print('scores', scores.shape)
    print('labels', labels.shape)

    # pred_y = tf.reduce_sum(probs * scores, 1)
    pred_y = tf.reduce_sum(logits, 1)

    diff = tf.subtract(labels, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))


def softmax_cross_entropy(outputs, labels):
    """ computes average softmax cross entropy """

    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    print('outputs:', outputs.shape)
    print('labels:',labels.shape)
    pred_y = tf.reduce_sum(outputs, 1)
    loss = tf.square(tf.subtract(pred_y, labels)) * 0.5
    return tf.reduce_mean(loss)

def HitRate(outputs, lables, test_u_indices, type='rt', topK=17):
    if type=='rt':
        cc = 0.1
    else:
        cc = 5
    result = {}
    for i in range(len(test_u_indices)):
        if test_u_indices[i] in result.keys():
            result[test_u_indices[i]].append((i, outputs[i]))
        else:
            result[test_u_indices[i]] = [(i, outputs[i])]
    for i in result.keys():
        result[i].sort(key=lambda x:x[1])
    label = []
    prediction = []
    index = []
    for i in result.keys():
        count = 0
        for j in result[i]:
            if count > (topK-1):
                break
            label.append(lables[j[0]])
            prediction.append(j[1])
            index.append(j[0])
            count += 1
    hit = np.abs(np.array(prediction).reshape((-1)) - np.array(label))
    one = np.ones_like(hit)
    zero = np.zeros_like(hit)
    ifHit = np.where(hit <= cc, one, zero)
    print('hitRate_ifHit',ifHit.shape)
    hitRate = np.mean(ifHit)
    return hitRate

def Diversity(outputs, lables, test_u_indices, type='rt', topK=17):
    if type=='rt':
        cc = 0.1
    else:
        cc = 5
    result = {}
    for i in range(len(test_u_indices)):
        if test_u_indices[i] in result.keys():
            result[test_u_indices[i]].append((i, outputs[i]))
        else:
            result[test_u_indices[i]] = [(i, outputs[i])]
    for i in result.keys():
        result[i].sort(key=lambda x:x[1])
    label = []
    prediction = []
    index = []
    userIndex = []
    for i in result.keys():
        count = 0
        for j in result[i]:
            if count > (topK-1):
                break
            label.append(lables[j[0]])
            prediction.append(j[1])
            index.append(j[0])
            userIndex.append(i)
            count += 1
    hit = np.abs(np.array(prediction).reshape((-1)) - np.array(label))
    one = np.ones_like(hit)
    zero = np.zeros_like(hit)
    ifHit = np.where(hit <= cc, one, zero)
    print('ifHit:', len(ifHit))
    print('predection', len(prediction))
    print(ifHit.shape)
    dele = np.where(ifHit == 0)
    print('dele', len(dele[0]))
    userIndex = np.delete(userIndex, dele[0])
    prediction = np.delete(prediction, dele[0])
    print('predection', len(prediction))
    diversity_dict = {}
    for i in range(len(userIndex)):
        if userIndex[i] in diversity_dict.keys():
            diversity_dict[userIndex[i]].append((i, prediction[i]))
        else:
            diversity_dict[userIndex[i]] = [(i, prediction[i])]
    sum_diver = []
    for i in diversity_dict.keys():
        sum_d = 0
        for j in range(len(diversity_dict[i])):
            temp = diversity_dict[i][j]
            for k in range(j+1,len(diversity_dict[i])):
                a = np.sqrt(np.sum(np.asarray(temp[1]-diversity_dict[i][k][1]) ** 2, axis=0))
                sum_d += a
        if len(diversity_dict[i]) > 1:
            sum_d = sum_d/(len(diversity_dict[i])*(len(diversity_dict[i])-1)/2)
        elif len(diversity_dict[i]) == 0:
            sum_d = 0
        sum_diver.append((1-sum_d))

    return np.mean(sum_diver)