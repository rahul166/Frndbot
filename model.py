##This model is basic model and need parameter tuning
##prediction task remaning

from keras.models import Model
from keras.layers import Input, LSTM, Dense


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.


latent_dim=256
encoder_input_data=X[:,0,:,:]
print(encoder_input_data.shape)
# print(encoder_input_data)


encoder_inputs = Input(shape=(None,73 ))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, 73))


decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(73, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, y], decode_target,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)



##################################################

####Prediction Task#################################

