def CNN_Model(data, labels, CNN_Filters = 3, Split = .2, Class_Nr = 20, EPCH = 10):

  # I assume that data has 4D, (sample Nr, image length, image width, color channels)
  # Split --> which fraction of dataset is taken as test and train 
  # Class_Nr --> number of classes 
  # EPCH --> epoch numbers

  from keras import backend as K
  K.clear_session()

  s, x, y, z = data.shape
  data = data / np.max(data)
  # randomizing data order
  #Data = data[np.random.permutation(s)]
  SP = np.int(np.floor(s * (1- Split)))
  train_data = data[0:SP]
  test_data = data[SP+1:s]
  train_labels = labels[0:SP]
  test_labels = labels[SP+1:s]
  #pdb.set_trace()

  from keras.utils import to_categorical
  #one-hot encode target column
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)


  from keras.models import Sequential
  from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
  #create model
  model = Sequential()
  #add model layers
  model.add(Conv2D(32, kernel_size=CNN_Filters, activation='relu', input_shape=(x,y,z)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, kernel_size=CNN_Filters, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=CNN_Filters, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))


  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(Class_Nr, activation='softmax'))

  model.summary()

  #compile model using accuracy to measure model performance
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


  #train the model
  model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=EPCH, shuffle=True)

  return model
    