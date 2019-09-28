import tensorflow as tf
import numpy as np

my_graph = tf.Graph()
with my_graph.as_default():
    a1 = tf.constant(np.ones([4, 4]))
    a2 = tf.constant(np.ones([4, 4]))
    a1_dot_a2 = tf.matmul(a1, a2)

    b1 = tf.Variable(a1)
    b2 = tf.Variable(np.ones([4, 4]))
    a1_elementwise_a2 = a1 * a2
    a1_dot_a2 = tf.matmul(a1, a2)
    b1_dot_b2 = tf.matmul(b1, b2)

init = tf.global_variables_initializer()
tf.summary.FileWriter("a1dota2", graph=my_graph)
sess = tf.Session(graph=my_graph)
print(sess.run(a1_dot_a2))
print(sess.run(init))
