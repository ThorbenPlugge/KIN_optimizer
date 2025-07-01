import tensorflow as tf
import time

@tf.function
def heavy_func(x):
    y = x
    for _ in range(100):
        y = y * x + tf.sin(y)
    return y

start = time.time()
heavy_func(tf.constant(2.0))
print("Graph building time:", time.time() - start)