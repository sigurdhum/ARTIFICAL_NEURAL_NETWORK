import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def show_images(images, labels):
    for y in range(len(images) // 25):
            plt.title("Pictures")
            for x in range(25):
                plt.subplot(5, 5, x + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[x + y * 25].astype('uint8'))
                plt.xlabel(str(x + y * 25) + " " + labels[x + y * 25])
            plt.show()


class Model:
    def __init__(self, path):
        #Model should have 3 classes, healthy, unhealthy and unknown
        dataset = np.load(os.path.join(path, 'public_data.npz'), allow_pickle=True)
        self.images, self.labels = dataset['data'], dataset['labels']

        
        
        unknownIDs_some = [58, 94, 95, 137, 138, 171, 207, 338, 412, 434, 486, 506, 429, 516, 571, 599, 622, 658]

        #have the unknowns in the dataset into a seperate array
        unknowns = []
        unknownIDs = []
        deleted = 0
        for i in range(len(unknownIDs_some)):
            unknowns.append(self.images[unknownIDs_some[i]-deleted])
            unknownIDs.append(self.labels[unknownIDs_some[i]-deleted])
            self.images = np.delete(self.images, unknownIDs_some[i]-deleted, 0)
            self.labels = np.delete(self.labels, unknownIDs_some[i]-deleted, 0)
            deleted += 1

        #show_images(self.images, self.labels)

        train_unknown_dataset = tf.data.Dataset.from_tensor_slices((unknowns, unknownIDs))

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(96, 96)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['sparse_categorical_accuracy'])

        model.fit(train_unknown_dataset, epochs=10)
        



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
    m = Model("")
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
