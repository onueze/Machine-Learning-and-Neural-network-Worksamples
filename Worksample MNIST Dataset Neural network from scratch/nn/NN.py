import numpy as np
from typing import List, Optional, Dict, Tuple

from .layers import Layer
from .losses import Loss
from .layers import FullyConnectedLayer
from .utils import Utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class NN:
    """
    Neural Network Class.
    """
    
    

    layers: List[Layer]

    loss: Loss = None

    lr: float = 0.001
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0

    best_weights_biases: Dict[int, Tuple[np.array, np.array]] = None
    min_loss: Tuple[int, float] = (None, float('inf'))

    def __init__(self, layers: Optional[List[Layer]] = None):
        """
        Constructs a Neural Network using pre-layered architecture.
        :param layers: list of Layers. Can be of Activation, Dropout or Fully Connected type.
        """
        self.testing = False

        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def _set_params(self, lr: float, l1_lambda: float, l2_lambda: float, momentum_rate: float):
        """
        Sets the learning rate, L1 and L2 coefficients for the Fully Connected layers in the network.
        :param lr: float learning rate.
        :param l1_lambda: float L1 coefficient.
        :param l2_lambda: float L2 coefficient.
        :param momentum_rate: float momentum_rate coefficient.
        :return:
        """

        self.lr = lr
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.momentum_rate = momentum_rate

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, FullyConnectedLayer):
                layer.lr = self.lr
                layer.l1_lambda = self.l1_lambda
                layer.l2_lambda = self.l2_lambda
                layer.momentum_rate = self.momentum_rate

    def add_layer(self, layer: Layer):
        """
        Adds a layer to the network.
        :param layer: Layer. Can be of Activation, Dropout or Fully Connected type.
        :return:
        """

        self.layers.append(layer)

    def _forward(self, x: np.array):
        """
        Forward-passes the data through the network.
        :param x: input data.
        :return:
        """

        for layer in self.layers:
            if(self.testing):
                layer.training = False
                x = layer.forward(x)
            else:
                layer.training = True
                x = layer.forward(x)

        return x

    def _backward(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Backward-passes the data through the network.
        :param y_pred: network's output.
        :param y_true: true value.
        :return:
        """

        dE_dY = self.loss.derivative(y_pred=y_pred, y_true=y_true)
        for layer in reversed(self.layers):
            dE_dY = layer.backprop(dE_dY)
        return dE_dY

    def _reload_weights(self, best_weights: Dict[int, Tuple[np.array, np.array]]):
        """
        Reload the weights and biases of the network from the best_weights.
        :param best_weights: Dictionary containing the saved weights and biases.
        """

        for i, layer in enumerate(self.layers):
            if i in best_weights and isinstance(layer, FullyConnectedLayer):
                layer.weights, layer.biases = best_weights[i]

    def fit(self,
            x: np.array,
            y: np.array,
            loss_function: Loss,
            epochs: int,
            batch_size: int,
            lr: float = 0.01,
            l1_lambda: float = 0.0,
            l2_lambda: float = 0.0,
            momentum_rate: float = 0.0,
            select_minimum_loss_after_training: bool = True,
            early_stopping: bool = True,
            patience: int = 2,
            verbose: bool = True,
            validation_split: float = 0.2
            ) -> Tuple[Dict[int, float], Tuple[int, float]]:
        """
        Initiates the training process on the given data.

        :param x: numpy array of the features.
        :param y: numpy array of the target values.
        :param loss_function: loss function to measure the network's performance.
        :param epochs: determines the number of Forward-Backward cycles for training.
        :param batch_size: number of samples to be processed during one Forward-Backward cycle.
        :param lr: intensity, at which the fully-connected layers will update its weights. If the value was already set, it will be overriden.
        :param l1_lambda: L1 regularization coefficient for the fully-connected layers. If the value was already set, it will be overriden.
        :param l2_lambda: L2 regularization coefficient for the fully-connected layers. If the value was already set, it will be overriden.
        :param momentum_rate: rate at which the velocity contributes to the gradient update. 0.9 is default.
        :param select_minimum_loss_after_training: Variant of early-stopping. False by default. if True is passed, then after training the NN would "reload" the weights from the epoch which had the minimum loss throughout the training.
        :param early_stopping: Boolean that indicates when neural network stops the training process due to not showing improvement over a certain amount of epochs.
        :param patience: if over a number of epochs no improvement is shown, the neural network will stop training
        :param verbose: ability to see the network's performance over training.
        :param validation_split: Percentage of data to be used for validation
        :return: 2-Tuple of [dictionary of losses at each of the epochs | 2-tuple with the epoch with min. loss]
        """

        self.loss = loss_function
        self._set_params(lr=lr, l1_lambda=l1_lambda, l2_lambda=l2_lambda, momentum_rate=momentum_rate)
        
        loss_dict = {}
        validation_loss_dict = {}
        accuracy_dict = {}
        val_accuracy_dict = {}
        
        no_improvement_count = 0
        best_loss = np.inf
        best_loss_val = np.inf
        
        num_samples = x.shape[0]
        num_validation = int(validation_split * num_samples)
        num_train = num_samples - num_validation

        for ep in range(epochs):
            total_loss = 0
            total_correct = 0
            num_batches = 0
            
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x_train = x[indices]
            y_train = y[indices]
            
            if validation_split > 0:
                x_train, x_val = x_train[:num_train], x_train[num_train:]
                y_train, y_val = y_train[:num_train], y_train[num_train:]

            for start_idx in range(0, len(x_train), batch_size):
                end_idx = min(start_idx + batch_size, len(x_train))
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                y_pred = self._forward(x_batch)
                self._backward(y_pred, y_batch)

                batch_loss = self.loss.loss(y_pred=y_pred, y_true=y_batch)

                total_loss += np.mean(batch_loss)
                num_batches += 1

                predicted_labels = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                total_correct += np.sum(predicted_labels == true_labels)

            average_loss = (total_loss / num_batches)

            if select_minimum_loss_after_training and average_loss < best_loss:
                best_loss = average_loss
                self.min_loss = (ep + 1, best_loss)
                self.best_weights_biases = {i: (layer.weights.copy(), layer.biases.copy()) for i, layer in
                                            enumerate(self.layers) if
                                            isinstance(layer, FullyConnectedLayer)}
                
            

            loss_dict[ep] = average_loss
            accuracy = total_correct / num_train
            accuracy_dict[ep] = accuracy
            
            
            if validation_split > 0:
                # ratio between validation size and training size to scale batch size for validation evaluation
                ratio_train_val = num_train / num_validation
                val_batch_size = int(batch_size / ratio_train_val)

            
            # Evaluate on validation set
            if validation_split > 0:
                val_loss, total_correct_val = self._evaluate_validation(x_val, y_val, batch_size=batch_size)
                validation_loss_dict[ep] = val_loss
                
                if select_minimum_loss_after_training and val_loss < best_loss_val:
                    best_loss_val = val_loss
                    self.min_loss = (ep + 1, best_loss_val)
                    self.best_weights_biases = {i: (layer.weights.copy(), layer.biases.copy()) for i, layer in
                                                enumerate(self.layers) if
                                                isinstance(layer, FullyConnectedLayer)}
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # adding accuracy to dictionary
                accuracy_val = total_correct_val / num_validation
                val_accuracy_dict[ep] = accuracy_val
                

            if verbose:
                print(f"{Utils.OKBLUE}{Utils.BOLD}\nEpoch {ep + 1}/{epochs}{Utils.ENDC}")
                print(f"{Utils.OKGREEN}Training Average Loss: {average_loss:.6f}{Utils.ENDC}")
                if validation_split > 0:
                    print(f"{Utils.OKGREEN}Validation Averge Loss: {val_loss:.6f}{Utils.ENDC}")
                    print(f"{Utils.WARNING}Accuracy Validation: {accuracy_val * 100:.2f}%{Utils.ENDC}")
                print(f"{Utils.WARNING}Accuracy Train: {accuracy * 100:.2f}%{Utils.ENDC}")
                print(f"{Utils.BOLD}{Utils.OKBLUE}No Improvement Count: {no_improvement_count} | Patience to reach for early stop: {patience}{Utils.ENDC}")
                
            if early_stopping and no_improvement_count >= patience:
                if verbose:
                    print(f"{Utils.BOLD}{Utils.OKBLUE}\nEarly stopping after {patience} epochs with no improvement.{Utils.ENDC}")
                break
            
        
        if select_minimum_loss_after_training and self.best_weights_biases is not None:
            if verbose:
                print(f"{Utils.BOLD}{Utils.OKGREEN}\nLoading the least-loss weights and biases configuration from epoch {self.min_loss[0]} with loss {self.min_loss[1]:.4f}.{Utils.ENDC}")

            self._reload_weights(self.best_weights_biases)
        

        for layer in self.layers:
            layer.training = False

        if verbose:
            print(f"{Utils.OKBLUE}{Utils.BOLD}\nTraining is completed.{Utils.ENDC}")
            
        # plots
        plot_accuracy(accuracy_dict, val_accuracy_dict)
        plot_loss_curve(loss_dict,validation_loss_dict)
        plot_image_grid(x, y, self)
        
        return loss_dict, self.min_loss

    def predict(self, x: np.array) -> np.array:
        """
        Perform a forward pass using the input data and return the network's output.

        :param x: numpy array of input data.
        :return: numpy array of the network's predictions.
        """

        predictions = self._forward(x)

        return predictions
    
    def _evaluate_validation(self, x_val, y_val, batch_size=None):
        total_val_loss = 0
        num_val_batches = 0
        total_correct_val = 0
        self.testing = True
        
        
        if batch_size is None:
            # Process the entire validation set without batching
            x_val_batch = x_val
            y_val_batch = y_val
            
            
            y_val_pred = self._forward(x_val_batch)
            val_batch_loss = self.loss.loss(y_pred=y_val_pred, y_true=y_val_batch)

            predicted_labels = np.argmax(y_val_pred, axis=1)
            true_labels = np.argmax(y_val_batch, axis=1)
            total_correct_val += np.sum(predicted_labels == true_labels)

            total_val_loss += np.mean(val_batch_loss)
            num_val_batches += 1
        else:
            for val_start_idx in range(0, len(x_val), batch_size):
                val_end_idx = min(val_start_idx + batch_size, len(x_val))
                x_val_batch = x_val[val_start_idx:val_end_idx]
                y_val_batch = y_val[val_start_idx:val_end_idx]

                y_val_pred = self._forward(x_val_batch)
                val_batch_loss = self.loss.loss(y_pred=y_val_pred, y_true=y_val_batch)
            
                predicted_labels = np.argmax(y_val_pred, axis=1)
                true_labels = np.argmax(y_val_batch, axis=1)
                total_correct_val += np.sum(predicted_labels == true_labels)

                total_val_loss += np.mean(val_batch_loss)
                num_val_batches += 1

        average_val_loss = (total_val_loss / num_val_batches)
        
        self.testing = False
        return average_val_loss, total_correct_val
    
    
    
    
    def _calculate_confusion_matrix(self, x_test, y_test):
        y_pred = np.argmax(self.predict(x_test), axis=1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        classes = np.unique(y_test)

        # Plot confusion matrix
        def plot_confusion_matrix(conf_matrix, classes):
            plt.figure(figsize=(8, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="hot", xticklabels=classes, yticklabels=classes)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()
        plot_confusion_matrix(conf_matrix, classes=classes)
        

def plot_image_grid(x, y, model, num_rows=3, num_cols=3, font_size=8):
    # scatter plot with a perfect prediction line
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(x), axis=1)

    # Display MNIST images in a grid along with true and predicted labels
    num_images_to_display = num_rows * num_cols

    selected_indices = np.random.choice(len(x), num_images_to_display, replace=False)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8))

    for i in range(num_rows):
        for j in range(num_cols):
            idx = selected_indices[i * num_cols + j]
            image = x[idx].reshape(28, 28)
            true_label = y_true[idx]
            pred_label = y_pred[idx]

            axs[i, j].imshow(image, cmap='gray')
            axs[i, j].set_title(f'True Label: {true_label}, Predicted Label: {pred_label}', fontsize=font_size)
            axs[i, j].axis('off')

    plt.show()
    



def plot_accuracy(train_accuracy_dict, val_accuracy_dict):
    # Plot the loss
    plt.plot(list(train_accuracy_dict.keys()), list(train_accuracy_dict.values()), label='Training Accuracy')
    plt.plot(list(val_accuracy_dict.keys()), list(val_accuracy_dict.values()), label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_loss_curve(train_loss_dict, val_loss_dict):
    # Plot the loss
    plt.plot(list(train_loss_dict.keys()), list(train_loss_dict.values()), label='Training Loss')
    plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()