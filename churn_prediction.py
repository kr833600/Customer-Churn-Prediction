import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# Remove columns that are not useful for prediction
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Check the majority-class baseline
baseline = df["Exited"].value_counts(normalize=True).max()
print(f"Majority-class baseline: {baseline:.1%}")

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df, columns=["Geography", "Gender"], drop_first=True
)

# Separate input features and target
X = df.drop(columns=["Exited"]).astype(float)
y = df["Exited"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(11, activation="relu"),
    Dense(11, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile and train the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Make predictions
proba = model.predict(X_test).ravel()
pred = (proba > 0.5).astype(int)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))
