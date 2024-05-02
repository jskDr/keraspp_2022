"""
TensorFlow Example 1-1
"""
import streamlit as st
import tensorflow as tf

# 1. Define training data
@tf.function
def f(w: tf.Variable, b: tf.Variable, x: tf.Tensor) -> tf.Tensor:
    return w * x + b

wo = tf.constant(3.0)
bo = tf.constant(1.0)
x = tf.constant([1.,2.])
y = f(wo, bo, x)

st.header("TensorFlow Example 1-1")
st.subheader("Data and Model Preparation")

st.write(f"Model: $f(x) = {wo} x + {bo}$")
st.write(f"Input: $x = {x}$")
st.write(f"Output: $y = f(x) = {y}$")

# 2. Define model
w = tf.Variable(1.0)
b = tf.Variable(0.0)

y_pred = f(w, b, x)
st.write(f"Prediction: ${y_pred}$")

@tf.function
def loss(w: tf.Variable, b: tf.Variable, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    y_pred = f(w, b, x)
    return tf.reduce_mean(tf.square(y - y_pred))

l = loss(w, b, x, y)
st.write(f"Intial Loss: $J(w,b) = E|f(w_o, y_o, x) - f(w, y, x)|^2 = {l}$")

optimizer = tf.optimizers.SGD(learning_rate=0.1)

def train_step(w, b, x, y):
    with tf.GradientTape() as tape:
        l = loss(w, b, x, y)
    dw, db = tape.gradient(l, [w, b])
    optimizer.apply_gradients(zip([dw, db], [w, b]))

with st.spinner("Training..."):
    epochs = 1000
    for e in range(epochs):
        train_step(w, b, x, y)
        
st.subheader("Training Results")
x_test = tf.constant([3., 4., 5.])
y_test = f(wo, bo, x_test)
y_pred = f(w, b, x_test)
errors = y_test - y_pred

l_test = loss(w, b, x_test, y_test)
st.write(f"Final Test Loss: $J(w,b) = {l_test}$")
st.write(f"Test: $x = {x_test}$, $y = {y_test}$, $y_p = {y_pred}$")
st.write(f"Errors: {errors}")
