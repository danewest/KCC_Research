import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import inspect

print(f"Scikit-learn version being used: {sklearn.__version__}")
print(f"Function 'mean_squared_error' imported from: {mean_squared_error.__module__}")
print(f"Function 'mean_squared_error' signature: {inspect.getfullargspec(mean_squared_error)}")