#
# Based on the Tensorflow Guide
# A Guide to TF Layers: Building a Convolutional Neural Network
# https://www.tensorflow.org/tutorials/layers
#

from skimage.io import imsave
from skimage.color import gray2rgb
import tensorflow as tf
import numpy as np



lr = 0.01
batch_sz = 50
stps = 50000
output_name = './spiralimage.jpg'

def net(features, labels, mode):

    #input
    input_layer = tf.reshape(features["x"], [-1, 2])

    #dense layer
    dense1 = tf.layers.dense(inputs=input_layer, units=40, activation=tf.nn.selu)

    #dense layer
    dense2 = tf.layers.dense(inputs=dense1, units=40, activation=tf.nn.selu)
    
    dense3 = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.selu)
    
    #dense4 = tf.layers.dense(inputs=dense3, units=10, activation=tf.nn.sigmoid)

    #output layer
    output = tf.layers.dense(inputs=dense3, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=output, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(output, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
 #       print("hellohello")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
#    print("nothing")
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
 
 
def main(unused):
    #find total number of training values
    x=0
    print("processing input file")
    points_file = open("points.txt", "r")
    for line in points_file:
        x += 1
    
    #make input and labels arrays
    feat = np.zeros((x, 2))
    labels = np.zeros(x)
    
    #get data out of text file
    i=0
    points_file.close()

    points_file = open("points.txt", "r")
    for line in points_file:
        temp = line.split()
        #print(float(temp[0]))
        feat[i][0] = float(temp[0])
        feat[i][1] = float(temp[1])
        labels[i] = float(temp[2])
        i += 1

    print("file processed")
    
    
    feat = feat
    labels = labels.astype(dtype='int32')

    #print(feat)
    #print(labels)
    
    
    classifier = tf.estimator.Estimator(model_fn=net)#, model_dir = "model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    print("training")
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": feat},
        y=labels,
        batch_size=batch_sz,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=stps,
        hooks=[logging_hook])
    print("trained")

    print("testing accuracy")
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": feat},
        y=labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=test_input_fn)
    print(eval_results)


    scale = 80 #use an even number
    #set up inputs for image
    width = int(6.5*scale)
    image = np.zeros((width, width))
    row = 0
    image_feat = np.zeros((width*width, 2))
    for i in range(width):
        for j in range(width):
            image_feat[row][0] = (i/(6.5*scale))*13-6.5
            image_feat[row][1] = (j/(6.5*scale))*13-6.5
            row += 1

    print("creating image")
    second_test_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": image_feat},
        y=None,
        num_epochs=1,
        shuffle=False)
    pred = classifier.predict(input_fn=second_test_input)
    for i in range(width*width):
        cls = next(pred)['probabilities'][1]
#        print('x:', i//65, 'y:', i%65, 'class:', cls)
        image[i//width][i%width] = cls*255

    image = image/255
    img_rgb = gray2rgb(image)
    imsave(output_name, img_rgb)

if __name__ == "__main__":
    tf.app.run()
