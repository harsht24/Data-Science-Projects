from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle


def iris_model():
    iris_df = load_iris()

    x = iris_df.data
    y = iris_df.target

    model = LogisticRegression().fit(x, y)

    pickle.dump(model, open('model.pkl', 'wb'))

    # model = pickle.load(open('model.pkl', 'rb'))


iris_model()
