from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import MySQLdb

app = Flask(__name__)

# Connect to MySQL Database
db = MySQLdb.connect(host="localhost", user="root", passwd="root", db="thyroid_db", charset="utf8mb4")
cursor = db.cursor()

# Load model results
results_df = pd.read_csv("model_results.csv")

# Get the best model (highest F1-score)
best_model = results_df.loc[results_df["f1_score"].idxmax(), "model"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form["age"]
        sex = request.form["sex"]
        on_thyroxine = request.form["on_thyroxine"]
        sick = request.form["sick"]
        pregnant = request.form["pregnant"]
        goitre = request.form["goitre"]
        tumor = request.form["tumor"]

        # Convert inputs for the model
        input_data = np.array([[int(age), int(sex == "female"), int(on_thyroxine == "t"),
                                 int(sick == "t"), int(pregnant == "t"), int(goitre == "t"), int(tumor == "t")]])

        # Load the best model
        model = joblib.load(f"models/{best_model}.pkl")

        # Make a prediction
        prediction = model.predict(input_data)[0]
        result = "Positive" if prediction == 1 else "Negative"

        # Store data in MySQL database
        sql = "INSERT INTO dat (age, sex, on_thyroxine, sick, pregnant, goitre, tumor) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (age, sex, on_thyroxine, sick, pregnant, goitre, tumor)

        try:
            cursor.execute(sql, values)
            db.commit()
        except Exception as e:
            db.rollback()
            print("Error:", e)

        return render_template("result.html", prediction=result, results=results_df.to_dict(orient="records"), best_model=best_model)

    return render_template("index.html")

# Route to fetch stored data and display in view.html
@app.route("/view")
def view_data():
    cursor.execute("SELECT * FROM dat")  # Replace 'dat' with your actual table
    data = cursor.fetchall()
    return render_template("view.html", data=data)

# Chatbot function
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you with thyroid prediction?"
    elif "what is thyroid" in user_input:
        response = "The thyroid is a small butterfly-shaped gland in your neck that controls metabolism, energy levels, and hormone balance."
    elif "what is hypothyroidism" in user_input:
        return "Hypothyroidism is a condition where the thyroid gland does not produce enough hormones."
    elif "causes of hypothyroidism" in user_input:
        response = "Common causes include iodine deficiency, autoimmune diseases (Hashimoto's thyroiditis), and certain medications."
    elif "symptoms of hypothyroidism" in user_input:
        return "Fatigue, weight gain, dry skin, hair loss, and depression."
    elif "treatment for hypothyroidism" in user_input:
        response = "The most common treatment is daily thyroid hormone replacement therapy (Levothyroxine)."
    elif "what is hyperthyroidism" in user_input:
        response = "Hyperthyroidism occurs when the thyroid produces too much hormone, leading to weight loss, nervousness, and rapid heartbeat."
    elif "causes of hyperthyroidism" in user_input:
        response = "Common causes include Graves' disease, toxic nodular goiter, and excessive iodine intake."
    elif "symptoms of hyperthyroidism" in user_input:
        response = "Symptoms include weight loss, increased heart rate, sweating, irritability, and tremors."
    elif "treatment for hyperthyroidism" in user_input:
        response = "Treatment includes anti-thyroid medications, radioactive iodine therapy, or surgery in severe cases."
    elif "thyroid diet" in user_input:
        response = "For hypothyroidism, eat iodine-rich foods like fish, eggs, and dairy. For hyperthyroidism, reduce iodine intake and avoid caffeine."
    elif "how is thyroid diagnosed" in user_input:
        response = "Thyroid conditions are diagnosed with blood tests measuring TSH, T3, and T4 hormone levels."
    elif "precautions for thyroid" in user_input:
        response = "To maintain thyroid health: \n1. Consume a balanced diet rich in iodine and selenium. \n2. Avoid excessive soy and processed foods. \n3. Get regular exercise. \n4. Manage stress to prevent hormonal imbalances."
    elif "stages of thyroid disease" in user_input:
        response = (
            "Thyroid disease has different stages: \n"
            "1. **Subclinical Hypothyroidism** - Mild elevation of TSH with normal T3/T4 levels.\n"
            "2. **Mild Hypothyroidism** - TSH slightly elevated, T3/T4 slightly reduced.\n"
            "3. **Severe Hypothyroidism** - Very high TSH, low T3/T4 causing severe symptoms.\n"
            "4. **Hyperthyroidism Stages** - Ranges from mild (increased T3/T4) to severe (thyroid storm, a medical emergency)."
        )

    elif "thyroid hormone levels based on age" in user_input:
        response = (
            "Thyroid hormone levels vary by age: \n"
            "🔹 **Newborns:** TSH: 0.7-15.2 mIU/L, T4: 9-19 ug/dL\n"
            "🔹 **Children:** TSH: 0.7-6.4 mIU/L, T4: 6-13 ug/dL\n"
            "🔹 **Adults (20-60 years):** TSH: 0.4-4.0 mIU/L, T4: 5-12 ug/dL\n"
            "🔹 **Older Adults (60+ years):** TSH: 0.5-5.5 mIU/L, T4: 4.5-11.5 ug/dL"
        )

    elif "prevention of thyroid problems" in user_input:
        response = (
            "To prevent thyroid issues: \n"
            "1. Ensure proper iodine intake (salt, seafood, dairy). \n"
            "2. Regular exercise to maintain metabolism. \n"
            "3. Avoid excess processed foods and sugary drinks. \n"
            "4. Get regular thyroid check-ups, especially if you have a family history."
        )

    elif "precautions" in user_input:
        return "Eat a balanced diet rich in iodine, exercise regularly, and avoid stress."
    elif "bye" in user_input:
        return "Goodbye! Stay healthy!"
    else:
        return "I'm sorry, I don't understand. Please ask about thyroid-related topics."

# Route for chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_message = request.form["message"]
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
