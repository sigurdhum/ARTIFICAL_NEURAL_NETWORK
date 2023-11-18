import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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

def show_images(images, labels, number = 25):
    for y in range(len(images) // number):
            plt.title("Pictures")
            for x in range(number):
                plt.subplot(5, 5, x + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[x + y * number].astype('uint8'))
                plt.xlabel(str(x + y * number) + " " + labels[x + y * number])
            plt.show()


class Model:
    def __init__(self, path):
        #Model should have 3 classes, healthy, unhealthy and unknown
        dataset = np.load(os.path.join(path, 'public_data.npz'), allow_pickle=True)
        self.images, self.labels = dataset['data'], dataset['labels']

        
        
        unknownIDs_some = [58, 338]


        # See the distribution of the data in the dataset before and after deleting the unknowns
        #self.pie_chart_labels(self.labels)
        self.images, self.labels =self.delete_unwanted(self.images, self.labels, unknownIDs_some)
        #self.pie_chart_labels(self.labels)

        #print number of unique images
        self.images, self.labels = self.uniques(self.images, self.labels)
        
        self.OG_images = self.images
        self.OG_labels = self.labels

        self.labels = np.where(self.labels == 'healthy', 0, 1)
        
        #self.pie_chart_labels(self.labels)

        split_idx = int(0.8 * len(self.images)) #TODO, make it a 50/50 split of healthy and unhealthy in both train and test
        
        self.train_images, self.test_images = self.images[:split_idx], self.images[split_idx:]
        self.train_labels, self.test_labels = self.labels[:split_idx], self.labels[split_idx:]

        #self.train_images = self.apply_data_augmentation(self.train_images)

        #self.pie_chart_labels(self.train_labels)
        #self.pie_chart_labels(self.test_labels)
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        
        self.make_model(path)



        #self.make_model(self, path)
    
    def apply_data_augmentation(self, images):
        augmented_images = []
        for image in images:
            # Random rotation (between -20 and 20 degrees)
            rotated_image = ndimage.rotate(image, np.random.uniform(-20, 20), reshape=False)
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                rotated_image = np.fliplr(rotated_image)
            
            # Resize the image to (96, 96, 3)
            resized_image = tf.image.resize(rotated_image, (96, 96))
            augmented_images.append(resized_image)
        return np.array(augmented_images)


    def uniques(self, images, labels):
        unique_images, unique_indices = np.unique(images, return_index=True, axis=0)
        unique_labels = labels[unique_indices]
        return unique_images, unique_labels
        
    def make_model(self, path):
        self.model = tf.keras.Sequential([
            #first convolution
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(96, 96, 3), padding='same'),
            #normalize the data
            tf.keras.layers.BatchNormalization(),
            #maxpooling layer
            tf.keras.layers.MaxPooling2D(2, 2),
            
            #second convolution
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(96, 96, 3), padding='same'),
            #normalize the data
            tf.keras.layers.BatchNormalization(),
            #maxpooling layer
            tf.keras.layers.MaxPooling2D(2, 2),

            #third convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(96, 96, 3), padding='same'),
            #normalize the data
            tf.keras.layers.BatchNormalization(),
            #maxpooling layer
            tf.keras.layers.MaxPooling2D(2, 2),
            
            #fourth convolution
            tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(96, 96, 3), padding='same'),
            #normalize the data
            tf.keras.layers.BatchNormalization(),
            #maxpooling layer
            tf.keras.layers.MaxPooling2D(2, 2),
            
            #fifth convolution
            tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(96, 96, 3), padding='same'),
            #normalize the data
            tf.keras.layers.BatchNormalization(),
            #maxpooling layer
            tf.keras.layers.MaxPooling2D(2, 2),

            #sixth convolution
            tf.keras.layers.Conv2D(512, (3,3), activation='relu', input_shape=(96, 96, 3), padding='same'),
            #normalize the data
            tf.keras.layers.BatchNormalization(),
            #maxpooling layer
            tf.keras.layers.MaxPooling2D(2, 2),

            #flatten the data
            tf.keras.layers.Flatten(),

            #dense layer and dropout
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.1),

            #dense layer and dropout
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            #dense layer and dropout
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),


            # Dense layer with 64 neurons and dropout
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),

            #output layer
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        

        total_samples = len(self.labels)
        healthy_samples = np.sum(self.labels == 0)
        unhealthy_samples = np.sum(self.labels == 1)

        weight_for_healthy = (1 / healthy_samples) * (total_samples / 2.0)
        weight_for_unhealthy = (1 / unhealthy_samples) * (total_samples / 2.0)

        class_weights = {0: weight_for_healthy, 1: weight_for_unhealthy}

        

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        
        self.checkpoint_path = path + "LEVERINGSMAPPE/" + "augemented_datagen_model"
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, verbose=1, save_best_only=True)

        #early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-12, restore_best_weights=True)

        #Lr reducer
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)

        #callbacks
        callbacks = [checkpointer, early_stopping, lr_reducer]

        batch_size = 32

        self.model.fit_generator(
            self.datagen.flow(self.train_images, self.train_labels, batch_size=batch_size),
            steps_per_epoch=len(self.train_images) // batch_size,
            validation_data=(self.test_images, self.test_labels),
            epochs=25,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights
        )


        #self.model.fit(self.images, self.labels, epochs=25, verbose = 1)

        # Save the weights
        #self.model.save_weights('path_to_save_weights.h5')
    
    def pie_chart_labels(self, labels):
        #pie chart
        unique, counts = np.unique(labels, return_counts=True)
        unique = np.where(unique == 0, "healthy", "unhealthy")
        plt.pie(counts, labels=unique, autopct='%1.1f%%')
        plt.title("Labels " + str(len(labels)))

        plt.show()
    
    def pie_chart_of_images(self, images):
        #pie chart
        for i in range(10):
            unique, counts = np.unique(images[i:i+10], return_counts=True)
            plt.pie(counts, labels=unique, autopct='%1.1f%%')
            plt.show()

    def delete_unwanted(self, images, labels, unwanted: list):
        #have the unknowns in the dataset into a seperate array
        unknowns = []
        unknownLabels = []
        deleted = 0
        for i in range(len(unwanted)):
            #get the image
            
            #if the image is in the unknowns, dont add it to the unknowns
            if not any(np.array_equal(images[unwanted[i]-deleted], unknown) for unknown in unknowns):
                blacklistedImage = images[unwanted[i]-deleted]
                unknowns.append(blacklistedImage)
                blacklistedLabel = labels[unwanted[i]-deleted]
                unknownLabels.append(blacklistedLabel)

            images = np.delete(images, unwanted[i]-deleted, 0)
            labels = np.delete(labels, unwanted[i]-deleted, 0)
            deleted += 1
        
        print("Deleted: ", len(unknowns))
        #show_images(unknowns, unknownLabels, number=max(len(unknowns), 25))

        deleted = 0
        for i in range(len(images)):
            index = i - deleted
            if any(np.array_equal(images[index], unknown) for unknown in unknowns):
                images = np.delete(images, index, 0)
                labels = np.delete(labels, index, 0)
                deleted += 1
        print("Deleted: ", deleted, "images")

        return images, labels

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
        self.model = tf.keras.models.load_model(self.checkpoint_path)
        predictions = self.model.predict(X)
        return tf.argmax(predictions, axis=1)

if __name__ == "__main__":
    m = Model("../")
    m.make_model("../")
    result = m.predict(m.images)
    #print out the first 10 predictions
    for i in range(10):
        print(str(int(result[i].numpy() == m.labels[i])), "Modellen:", result[i].numpy(), "Label:", m.labels[i], "Index:", i)

    #print out the accuracy
    print("Accuracy:")
    print(np.mean(result == m.labels), "% of " , len(m.labels), "correct")