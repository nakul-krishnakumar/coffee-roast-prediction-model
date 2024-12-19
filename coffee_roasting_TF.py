import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))
    
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))

X, Y = load_coffee_data();
print(X.shape, Y.shape)

print(f"Temperature Max, Min pre normalisation: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration Max, Min pre normalisation: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

# APPLYING NORMALIZATION
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X) # learns the mean and variance of the data set and saves the values internally.
Xn = norm_l(X)

print(f"Temperature Max, Min post normalisation: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration Max, Min post normalisation: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# REPLICATING DATA TO INCREASE PRECISION
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))
print(Xt.shape, Yt.shape)

# BUILDING THE MODEL
tf.random.set_seed(1234) # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)), # specifies the expected shape of the input.
        Dense(3, activation='sigmoid', name='layer1'),
        Dense(1, activation='sigmoid', name='layer2')
    ]
)

model.summary() # info on the model

# Examining weights W and bias b for both the layers
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# COMPILING THE MODEL
# defines a loss function and specifies a compile optimization.
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
)

# FITTING THE MODEL
# runs gradient descent and fits the weights to the data.
model.fit(
    Xt,Yt,
    epochs=10,
)
"""
    In the fit statement above, the number of epochs 
    was set to 10. This specifies that the entire 
    data set should be applied during training 10 times. 
"""

# UPDATED WEIGHTS AFTER FITTING
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)


# PREDICTING USING THE MODEL
X_test = np.array([
    [200, 13.9],
    [200, 17]
])

X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print('predictions = \n', predictions)

# MAKING DECISION
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0

print(f'decisions = \n{yhat}')

# alternative way to find yhat:
"""
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")
"""