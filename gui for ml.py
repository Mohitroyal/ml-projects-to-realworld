import tkinter as tk
from sklearn.linear_model import LinearRegression
import numpy as np

# Load or recreate your trained model
model = LinearRegression()
model.coef_ = np.array([[0.5]])
model.intercept_ = 40  
# Final Marks = 0.5 × Attendance + 40
# Function to predict
def predict_marks():
    attendance = float(entry.get())
    marks = model.predict(np.array([[attendance]]))
    result_label.config(text=f"Predicted Marks: {marks[0][0]:.2f}")


# GUI Setup
root = tk.Tk()
root.geometry("400x400")
root.config(bg = "gray")
root.title("Attendance to Marks Predictor")

tk.Label(root, text="Enter Attendance Hours:").pack()
entry = tk.Entry(root)
entry.pack()

tk.Button(root, text="Predict", command=predict_marks).pack()
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
