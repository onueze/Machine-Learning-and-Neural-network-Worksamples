import numpy as np
import pickle
from nn import NN, FullyConnectedLayer, ActivationLayer, DropoutLayer, Sigmoid, ReLU, Softmax, CategoricalCrossEntropy, Utils
import sys


def train():
    X_train_normalized, y_train, _, _ = Utils.prepareData()
    
    report_1_sigmoid = NN([
        FullyConnectedLayer(in_size=784, out_size=4),
        ActivationLayer(activation=Sigmoid()),
        FullyConnectedLayer(in_size=4, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])

    report_2_relu = NN([
        FullyConnectedLayer(in_size=784, out_size=128),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=128, out_size=64),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=64, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])
    
    report_3_relu = NN([
        FullyConnectedLayer(in_size=784, out_size=200),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.15),
        FullyConnectedLayer(in_size=200, out_size=400),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=400, out_size=300),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=300, out_size=200),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=200, out_size=100),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.25),
        FullyConnectedLayer(in_size=100, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])

    balanced_arch = NN([
        FullyConnectedLayer(in_size=784, out_size=300),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.15),
        FullyConnectedLayer(in_size=300, out_size=100),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.25),
        FullyConnectedLayer(in_size=100, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])
    
    # 0.9678 on test set (to achieve this set validation_split to 0 and lr= 0.001, batch size= 48, epochs=16)
    balanced_arch_with_dropout = NN([
        FullyConnectedLayer(in_size=784, out_size=300),
        ActivationLayer(activation=ReLU()),
        DropoutLayer(dropout_rate=0.15),
        FullyConnectedLayer(in_size=300, out_size=100),
        ActivationLayer(activation=ReLU()),
        DropoutLayer(dropout_rate=0.35),
        FullyConnectedLayer(in_size=100, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])
    
    
    # 0.9762 on test set with validation
    balanced_arch_with_dropout_new = NN([
        FullyConnectedLayer(in_size=784, out_size=300, l2_lambda=0.002),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=300, out_size=100, l1_lambda=0.001),
        ActivationLayer(activation=ReLU()),
        FullyConnectedLayer(in_size=100, out_size=100),
        ActivationLayer(activation=ReLU()),
        DropoutLayer(dropout_rate=0.2),
        FullyConnectedLayer(in_size=100, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])
    
    
    # better curve
    bugged_arch = NN([
        FullyConnectedLayer(in_size=784, out_size=1024),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.15),
        FullyConnectedLayer(in_size=1024, out_size=512),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.15),
        FullyConnectedLayer(in_size=512, out_size=256),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.15),
        FullyConnectedLayer(in_size=256, out_size=128),
        ActivationLayer(activation=ReLU()),
        # DropoutLayer(dropout_rate=0.4),
        FullyConnectedLayer(in_size=128, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])

    bugged_arch_2 = NN([
        FullyConnectedLayer(in_size=784, out_size=128),
        ActivationLayer(activation=ReLU()),
        DropoutLayer(dropout_rate=0.1),

        FullyConnectedLayer(in_size=128, out_size=64),
        ActivationLayer(activation=ReLU()),

        FullyConnectedLayer(in_size=64, out_size=32),
        ActivationLayer(activation=ReLU()),

        FullyConnectedLayer(in_size=32, out_size=10),
        ActivationLayer(activation=Softmax()),
    ])

    balanced_arch_with_dropout_new.fit(x=X_train_normalized,
                      y=Utils.to_one_hot(y_train.to_numpy(), 10),
                      epochs=100,
                      lr=0.0001,
                      l1_lambda=0.00,
                      l2_lambda=0.00,
                      batch_size=75,
                      momentum_rate=0.9,
                      patience=7,
                      validation_split=0.2,
                      early_stopping=True,
                      select_minimum_loss_after_training=True,
                      loss_function=CategoricalCrossEntropy())


    # FIT WHICHEVER
    return balanced_arch_with_dropout_new

def save_model(filename: str, trained_model: NN):
    with open(f"{filename}", 'wb') as file:
        pickle.dump(trained_model, file)


def test(xt: np.array, yt: np.array, model: NN):
    total = 0
    correct = 0
    incorrect_idx = []
    
    # Calculate confusion matrix
    confusion_matrix = model._calculate_confusion_matrix(xt, yt)
    
    for i in range(len(yt)):
        total += 1

        pred = np.argmax(model.predict(xt[i]))
        true = yt[i]

        if true == pred:
            correct += 1
        else:
            incorrect_idx.append(i)

    return (correct / total), incorrect_idx


def import_and_test(xt: np.array, yt: np.array, filename: str = "model.pkl"):
    with open(f"mnist_models/{filename}", 'rb') as f:
        loaded_model = pickle.load(f)
    if loaded_model is not None:
        print("Testing...\n")
        acc, incorrect_idx = test(xt, yt, loaded_model)
        print(f"Accuracy for {filename}: {acc}")
    else:
        print(f"\n{filename} was not found in mnist_models directory. Please try again")
        return


if __name__ == '__main__':

    if len(sys.argv) < 2:

        print("Incorrect usage. \n"
              "python3 mnist_tests.py [test | train] [pickle file name]")

    else:

        if sys.argv[1] == "test":
            if len(sys.argv) <= 2:
                print("Incorrect usage. \n"
                      "python3 mnist_tests.py test [pickle file name]")
            else:
                _, _, X_test_normalized, y_test = Utils.prepareData()
                import_and_test(xt=X_test_normalized, yt=y_test, filename=sys.argv[2])

        elif sys.argv[1] == "train":
            if len(sys.argv) <= 2:
                print("Incorrect usage. \n"
                      "python3 mnist_tests.py train [filename for new pickle model]")
            else:
                print("ok lets go")
                save_model(filename=sys.argv[2], trained_model=train())
