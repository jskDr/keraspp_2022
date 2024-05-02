import streamlit as st
import tensorflow as tf

st.title('Basic TensorFlow example 01')

x = tf.constant([1., 2.])
y = 2 * x + 1
