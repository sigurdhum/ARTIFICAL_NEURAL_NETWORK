import os
import tensorflow as tf
import numpy as np

'''

Dataset Details:

Image size: 96x96
Color space: RGB
File Format: npz
Number of classes: 2
Classes:
0: "healthy"
1: "unhealthy"
Dataset Structure

Single folder:
training: containing the 'public_data.npz' file. The file contains the following items:
'data': numpy array of shape 5100x96x96x3, containing the RGB images.
'data': 3-dimensional numpy array of shape 5200x96x96x3, containing the RGB images.
'labels': 1-dimensional numpy array of shape 5200 with values in {'healthy', 'unhealthy'}

'''


'''
__init__ function:

it is used to build the model and load the weights.
All the models that you want to load must be contained in the submission zip file,
at the same level of the script


Predict:

it uses the model to get the predictions.
This function must take the 4-dimensinal tensor (RGB images) and returns
the result of the interence.
The output must be the PREDICTED CLASSES and not the class probability
'''



class Model:
    def __init__(self, path):

        dataset = np.load(os.path.join(path, 'public_data.npz'), allow_pickle=True)
        self.images, self.labels = dataset['data'], dataset['labels']
        
        # Convert labels to integers
        self.labels = np.where(self.labels == 'healthy', 0, 1)

        split_idx = int(0.8 * len(self.images))
        self.train_images, self.val_images = self.images[:split_idx], self.images[split_idx:]
        self.train_labels, self.val_labels = self.labels[:split_idx], self.labels[split_idx:]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(96, 96, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            #second convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            #third convolution
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            #fourth convolution
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            #flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            #512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        
        # Compile the model
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        # Train the model
        self.model.fit(self.images, self.labels, epochs=25, verbose = 1)

        # Save the weights
        #self.model.save_weights('path_to_save_weights.h5')

        #Save the model
        self.model.save(path + "LEVERINGSMAPPE/" + 'modellen')
        self.modelpath = path + "LEVERINGSMAPPE/" + 'modellen' 

    def predict(self, X):
        '''
        predict():

        Params:
        -X: 
            4-dimensional input tensor. It contains a batch of images from the test set.
            Thus, X.shape = [BS, H, W, C], where BS is the batch size, H=96 is the image height,
            W=96 is the image width, and C=3 is the number of channels (RGB).
        Returns:
            predicted classes as 1-dimensional tensor of shape [BS]
        '''
        # Predict
        self.model = tf.keras.models.load_model(self.modelpath)
        predictions = self.model.predict(X)
        return tf.argmax(predictions, axis=1)

if __name__ == "__main__":
    m = Model("../")
    result = m.predict(m.images)
    #print out the first 10 predictions
    for i in range(10):
        if str(result[i].numpy()) != str(m.labels[i]):
            print("Modellen:")
            print(result[i].numpy())
            print("Fasit:")
            print(m.labels[i])

    #print out the accuracy
    print("Accuracy:")
    print(np.mean(result == m.labels), "% of " , len(m.labels), "correct")