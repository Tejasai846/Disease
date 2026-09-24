from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", symptoms=columns)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = [0] * len(columns)

    # Get checkbox values
    for i, col in enumerate(columns):
        if request.form.get(col) == "on":
            input_data[i] = 1

    # Optional textbox input (extra symptoms typed manually)
    extra = request.form.get("extra_symptoms")
    if extra:
        extra_list = [x.strip().lower() for x in extra.split(",")]

        for i, col in enumerate(columns):
            if col.lower() in extra_list:
                input_data[i] = 1

    # Prediction
    prediction = model.predict([input_data])

    return render_template("index.html",
                           symptoms=columns,
                           prediction_text="Predicted Disease: " + prediction[0])

if __name__ == "__main__":
    app.run(debug=True)