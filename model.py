from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(160,320,3)))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


data = Common.load_data('data')
data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
val_gen = DataGenerator(data_val, augment_data=False)
train_gen = DataGenerator(data_train, augment_data=True)

mode.fit_generator(train_gen,\
  samples_per_epoch=8192,\
  nb_epoch=100,\
  validation_data=val_gen,
  nb_val_samples=len(data_val),
  max_q_size=8192,
  nb_worker=8,
  pickle_safe=True)
