# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(#splitting the dataset
    X, y, test_size=0.2, random_state=42
)


#Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)#prediction


#Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

#Logistic Regression
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

# Evaluation
def evaluate(name, y_test, pred):
    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print()

evaluate("SVM Model", y_test, svm_pred)
evaluate("Decision Tree Model", y_test, tree_pred)
evaluate("Logistic Regression Model", y_test, log_pred)
