import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class BaseModel:
    
    def __init__(self, data,p_select):
        self.dataframe = data
        self.p_select = p_select
        self.model = None
        self.train_data = []
        self.test_data = []
        self.features = []
        self.labels = []
        self.split_data()
        self.train_data = self.transform_data_train(self.train_data)
        self.test_data = self.transform_data_test(self.test_data)
        
        
    
    def split_data(self,test_size = 0.2):
        """
        Split the data into training and testing sets.
        
        Parameters:
        - test_size: float, the proportion of the dataset to include in the test split (default is 0.2).
        """
        data_train, data_test = train_test_split(self.dataframe, test_size=test_size, random_state=42)
        self.train_data = data_train
        self.test_data = data_test.drop(columns=["util"])
        self.labels = data_test['util']
        print(f"Training data shape: {self.train_data.shape}, Testing data shape: {self.test_data.shape}")

    def transform_data_train(self,input_data):
        """
            Format the data properly to train the algorithm.
        """
        raise NotImplementedError("The transfom method should be implemented in subclasses.")

    def transform_data_test(self,input_data):
        """
                Format the data properly to train the algorithm.
        """
        raise NotImplementedError("The transfom method should be implemented in subclasses.")
    
    def train(self):
        """
        Train the model using the training data.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("The train method should be implemented in subclasses.")
    
    def predict(self, input_data) -> list[tuple[int, int]]:
        """
        Predict the output for the given input data.
        This method should be implemented in subclasses.
        
        Parameters:
        - input_data: The input data for which predictions are to be made.
        
        Returns:
        - The predicted output as a list of tuples
        First element of the tuple should contain the predicted classification in the binary case (0 or 1).
        Second element of the tuple should contain the predicted classification with the 4 classes (0, 1, 2, or 3). 
        """
        raise NotImplementedError("The predict method should be implemented in subclasses.")
    
    def evaluate(self,predictions):
        """
        Evaluate the model using the test data.
        This method should be implemented in subclasses.
        
        Returns:
        - The evaluation metrics (e.g., accuracy, loss).
        """
        # Calculate evaluation metrics (e.g., accuracy, loss)
        k=0
        for i in range(len(predictions)):
            if self.labels.iloc[i] == predictions[i]:
                k+=1
        some_metrics2 = k / len(predictions)
        some_metrics = some_metrics = accuracy_score(self.labels, predictions, normalize = True)  # Placeholder for actual evaluation logic
        print(f"Score avec sklearn: {some_metrics2}, score calculé {some_metrics} ")
        return some_metrics



if __name__ == "__main__":
    # Example usage
    model = BaseModel()  # Replace with actual data
