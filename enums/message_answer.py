from enum import Enum

# class syntax
class Message_Best (Enum):
    ACCURACY = "\nThe best accuracy is: "
    FUNCTION = "The activation function that obtained the best result is: "
    SOLVER = "The weight optimizer that obtained the best result is: "
    NUMBER_NEURONS = "The number of neurons in the hidden layer that obtained the best result is: "
    ACCURACY_SO_FAR = "The best accuracy found so far is:"

class Messages (Enum):
    ITERATIONS = "Number of iterations so far:"
    RUNNING_MODELS = "The models are still being tested"
