import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'modellen'))

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
                if len(labels) > 0:
                    blacklistedLabel = labels[unwanted[i]-deleted]
                    unknownLabels.append(blacklistedLabel)

            images = np.delete(images, unwanted[i]-deleted, 0)
            if len(labels) > 0:
                labels = np.delete(labels, unwanted[i]-deleted, 0)
            deleted += 1

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

        #delete the unwanted images
        #X, self.labels = self.delete_unwanted(X, [], unknownIDs_some)
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out
    
if __name__ == "__main__":
    m = model("")
    result = m.predict(np.load(os.path.join('../', "public_data.npz"), allow_pickle=True)["data"])
    labels = np.load(os.path.join('../', "public_data.npz"), allow_pickle=True)['labels']
    labels = np.where(labels == 'healthy', 0, 1)
    #print the accuracy
    print("Accuracy: ", np.sum(result == labels) / len(result))
    