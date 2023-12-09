import os
import numpy as np
from model import model


if __name__ == "__main__":
    m2 = model("./")
    m = model("./")
    
    print("er de like? ", m.model.summary() == m2.model.summary())
    readFromData = os.path.join("../", 'public_data.npz')
    predictions = m.predict(np.load(readFromData, allow_pickle=True)["data"])
    labels = np.load(readFromData, allow_pickle=True)["labels"]
    labels_as_int = np.where(labels == 'healthy', 0, 1)
    #for i in range(len(predictions)):
    #    print(str(int(predictions[i] == labels_as_int[i])) + "Prediction: ", predictions[i], "Label: ", labels[i])
    
    print("Accuracy:")
    print(np.mean(predictions == labels_as_int), "% of " , len(labels_as_int), "correct")