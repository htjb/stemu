import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from stemu.skutils import LogTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from stemu.emu import Emu

class CustomYTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Transform the dependent variable."""
        return X/ self.std
    
    def inverse_transform(self, X):
        """Inverse transform the dependent variable."""
        return X * self.std
    
    def fit(self, X):
        """Fit the transformer."""
        self.std = np.std(X)
        return self

class CustomXTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Transform the independent variable."""
        return (X - self.min) / (self.max - self.min)
    
    def inverse_transform(self, X):
        """Inverse transform the independent variable."""
        return X * (self.max - self.min) + self.min
    
    def fit(self, X):
        """Fit the transformer."""
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

# Load the data
X_train = np.loadtxt('downloaded_data/train_data.txt')[:2000]
X_test = np.loadtxt('downloaded_data/test_data.txt')[:100]
y_train = np.loadtxt('downloaded_data/train_labels.txt')[:2000]
y_test = np.loadtxt('downloaded_data/test_labels.txt')[:100]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# replace 0 with 1e-10
X_train[X_train == 0] = 1e-10
X_test[X_test == 0] = 1e-10

z = np.linspace(5, 50, y_train.shape[1])

print("z shape:", z.shape)

ypipe = Pipeline([('scalar', CustomYTransformer())])
xpipe = Pipeline([('log', LogTransformer(index=[0,1,2])), 
                  ('scalar', CustomXTransformer())])

emu = Emu(ypipe=ypipe, xpipe=xpipe)

emu.epochs = 50
emu.fit(X_train, z, y_train)

print(emu.history.__dict__)

pred_train = emu.predict(X_train)
pred_test = emu.predict(X_test)

train = [np.sqrt(np.mean((y_train[i] - pred_train[i])**2)) for i in range(len(y_train))]
test = [np.sqrt(np.mean((y_test[i] - pred_test[i])**2)) for i in range(len(y_test))]

plt.hist(train, bins=50, alpha=0.5, label='train', density=True)
plt.hist(test, bins=50, alpha=0.5, label='test', density=True)
plt.xlabel('RMSE [mK]')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of Errors')
plt.savefig('21cmGEM-histogram.png')
plt.close()

plt.scatter(y_train.flatten(), emu.predict(X_train).flatten(), alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Train)')
plt.savefig('21cmGEM-true_vs_predicted_train.png')
plt.close()

[plt.plot(z, y_test[i], c='k') for i in range(5)]
[plt.plot(z, emu.predict(X_test)[i], ls=':') for i in range(5)]
plt.xlabel('z')
plt.ylabel('Temperature')
plt.title('Predicted vs True Values (Test)')
plt.legend()
plt.savefig('21cmGEM-predicted_vs_true_test.png')
plt.close()

