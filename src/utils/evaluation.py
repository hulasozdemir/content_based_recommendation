from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Evaluation:
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k):
        recommended_at_k = recommended_items[:k]
        relevant_at_k = set(recommended_at_k) & set(relevant_items)
        return len(relevant_at_k) / k

    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k):
        recommended_at_k = recommended_items[:k]
        relevant_at_k = set(recommended_at_k) & set(relevant_items)
        return len(relevant_at_k) / len(relevant_items)

    @staticmethod
    def mean_absolute_error(predictions, targets):
        return mean_absolute_error(targets, predictions)

    @staticmethod
    def root_mean_squared_error(predictions, targets):
        return np.sqrt(mean_squared_error(targets, predictions))
