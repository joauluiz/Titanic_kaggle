from enum import Enum

# class syntax
class Message_Best (Enum):
    ACCURACY = "The best accuracy is: "
    BEST_FUNCTION = "The activation function that obtained the best result is: "
    BEST_SOLVER = "The weight optimizer that obtained the best result is: "
    BEST_NUMBER_NEURONS = "The number of neurons in the hidden layer that obtained the best result is: "

class Messages (Enum):
    ITERATIONS = "Number of iterations so far:"
    RUNNING_MODELS = "The models are still being tested"
    BEST_ACCURACY_SO_FAR = "\nThe best accuracy found so far is: "