import tensorflow as tf

def dice(yPred,yTruth,thresh):
    smooth = tf.constant(1.0)
    threshold = tf.constant(thresh)
    yPredThresh = tf.to_float(tf.greater_equal(yPred,threshold))
    mul = tf.mul(yPredThresh,yTruth)
    intersection = 2*tf.reduce_sum(mul) + smooth
    union = tf.reduce_sum(yPredThresh) + tf.reduce_sum(yTruth) + smooth
    dice = intersection/union
    return dice

if __name__ == "__main__":
    with tf.Session() as sess:

        print("Dice example")
        yP = tf.constant([0.1,0.9,0.7,0.3,0.1,0.1,0.9,0.9,0.1],shape=[3,3])
        yT = tf.constant([0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0],shape=[3,3])
        score = dice(yPred=yP,yTruth=yT,thresh= 0.5)
        scr ,pred, truth = sess.run([score,yP,yT])
        print(pred)
        print(truth)
        print(scr)
