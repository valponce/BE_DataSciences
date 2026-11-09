from matplotlib import pyplot as plt
import numpy as np

from base_model import BaseModel
from test import transform_train_data, transform_test_data
from sklearn import tree
import analyse_donnees as AN


class DecisionTree(BaseModel):
    def __init__(self, data,p_select):
        super().__init__(data,p_select)

    def transform_data_train(self,input_data):
        updated_data, self.features = transform_train_data(input_data,self.p_select)
        return updated_data

    def transform_data_test(self,input_data):
        updated_data = transform_test_data(input_data, self.features)
        return updated_data
    
    def train(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.train_data.drop(columns=['util']), self.train_data['util'])
        tree.plot_tree(clf)
        return clf

    def predict(self, model, input_data):
        #model.predict_data(input_data)
        return model.predict(input_data)


if __name__ == "__main__":
    # Example usage
    features_train = AN.read_ds("train")
    
    # Define the range of p values using numpy
    p_values = np.arange(0.1, 1.0, 0.1)
    accuracy_scores = []
    
    for p in p_values:
        # Initialize model with current p value (assuming your constructor takes it)
        model = DecisionTree(features_train, p) 
        clf = model.train()
        predictions = model.predict(clf, model.test_data)
        accuracy_scores.append(model.evaluate(predictions))

    # Plot using the p_values array for the x-axis
    plt.plot(p_values, accuracy_scores, marker='o')
    plt.xlabel("p value")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs. p-select")
    plt.grid(True)
    plt.savefig("accuracy_vs_p.png", dpi=300, bbox_inches='tight')
    plt.show()
