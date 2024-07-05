# Libraries
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Reading the file
data = pd.read_csv('./waterQuality1.csv')

# Droping rows that have invalid values
data = data.drop(index=[7551, 7568, 7890])

# Correcting the type of these columns data
data['ammonia'] = data['ammonia'].astype(float)
data['is_safe'] = data['is_safe'].astype(int)

# Spliting the train data from the test one
predictors_train, predictors_test, target_train, target_test = train_test_split(
    data.drop(['is_safe'], axis=1),
    data['is_safe'],
    test_size = 1/3,
    random_state = 123
)

# Training the model
model = DecisionTreeClassifier()
model = model.fit(predictors_train, target_train)