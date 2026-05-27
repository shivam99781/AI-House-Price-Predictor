# PREMIUM AI HOUSE PRICE PREDICTOR

#  IMPORTS 
import pandas as pd
import numpy as np
import customtkinter as ctk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import sqlite3

# APP SETTINGS

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# DATABASE

conn = sqlite3.connect("prediction_history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_price REAL
)
""")

conn.commit()

# LOAD DATASET

df = pd.read_csv("house_price_dataset_5000.csv")


# CLEANING

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# FEATURE ENGINEERING

df['luxury_score'] = (
    df['construction_quality_rating'] +
    df['neighborhood_quality_score'] +
    df['green_space_index']
) / 3


# ENCODING

df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

# FEATURES

features = [
    'property_area_sqft',
    'bedrooms',
    'bathrooms',
    'floors',
    'property_age',
    'distance_to_city_km',
    'neighborhood_quality_score',
    'construction_quality_rating',
    'luxury_score',
    'property_type_Villa'
]

X = df[features]
y = df['price']

# SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# SCALING

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = r2_score(y_test, pred)

# MAIN WINDOW

app = ctk.CTk()

app.geometry("1400x850")
app.title("AI House Price Prediction")

# SIDEBAR

sidebar = ctk.CTkFrame(
    app,
    width=280,
    corner_radius=0,
    fg_color="#111827"
)

sidebar.pack(side="left", fill="y")

logo = ctk.CTkLabel(
    sidebar,
    text="AI REAL ESTATE",
    font=("Arial", 30, "bold"),
    text_color="#00bfff"
)

logo.pack(pady=50)

acc = ctk.CTkLabel(
    sidebar,
    text=f"Model Accuracy\n{round(accuracy*100,2)}%",
    font=("Arial", 24, "bold"),
    text_color="lime"
)

acc.pack(pady=20)

line = ctk.CTkLabel(
    sidebar,
    text="━━━━━━━━━━━━━━",
    font=("Arial", 25),
    text_color="gray"
)

line.pack(pady=20)

info = ctk.CTkLabel(
    sidebar,
    text="AI Powered Property\nValuation System",
    font=("Arial", 18),
    text_color="white"
)

info.pack(pady=10)

# MAIN AREA

main = ctk.CTkScrollableFrame(
    app,
    fg_color="#0f172a"
)

main.pack(fill="both", expand=True)

# TITLE

title = ctk.CTkLabel(
    main,
    text="Premium AI House Price Predictor",
    font=("Arial", 42, "bold"),
    text_color="white"
)

title.pack(pady=30)

subtitle = ctk.CTkLabel(
    main,
    text="Enter Property Details",
    font=("Arial", 20),
    text_color="gray"
)

subtitle.pack(pady=5)

# INPUT CARD

card = ctk.CTkFrame(
    main,
    corner_radius=25,
    fg_color="#111827"
)

card.pack(pady=30, padx=30)

entries = {}

fields = [
    ("Property Area (sqft)", "property_area_sqft"),
    ("Bedrooms", "bedrooms"),
    ("Bathrooms", "bathrooms"),
    ("Floors", "floors"),
    ("Property Age", "property_age"),
    ("Distance To City (km)", "distance_to_city_km"),
    ("Neighborhood Score (1-10)", "neighborhood_quality_score"),
    ("Construction Quality (1-10)", "construction_quality_rating")
]


# CREATE INPUTS

for i, (label_text, key) in enumerate(fields):

    label = ctk.CTkLabel(
        card,
        text=label_text,
        font=("Arial", 18, "bold"),
        text_color="white"
    )

    label.grid(
        row=i,
        column=0,
        padx=30,
        pady=20,
        sticky="w"
    )

    entry = ctk.CTkEntry(
        card,
        width=350,
        height=50,
        corner_radius=15,
        font=("Arial", 18),
        border_width=2,
        border_color="#00bfff"
    )

    entry.grid(
        row=i,
        column=1,
        padx=30,
        pady=20
    )

    entries[key] = entry

# PROPERTY TYPE

ptype_label = ctk.CTkLabel(
    card,
    text="Property Type",
    font=("Arial", 18, "bold"),
    text_color="white"
)

ptype_label.grid(
    row=len(fields),
    column=0,
    padx=30,
    pady=20
)

ptype = ctk.CTkOptionMenu(
    card,
    values=["House", "Villa"],
    width=350,
    height=50,
    corner_radius=15,
    font=("Arial", 18),
    fg_color="#00bfff",
    button_color="#0099cc",
    button_hover_color="#0077aa"
)

ptype.grid(
    row=len(fields),
    column=1,
    padx=30,
    pady=20
)

# RESULT FRAME

result_frame = ctk.CTkFrame(
    main,
    height=250,
    corner_radius=30,
    fg_color="#111827"
)

result_frame.pack(
    fill="x",
    padx=40,
    pady=30
)

result_title = ctk.CTkLabel(
    result_frame,
    text="Predicted House Price",
    font=("Arial", 26, "bold"),
    text_color="gray"
)

result_title.pack(pady=20)

result = ctk.CTkLabel(
    result_frame,
    text="₹ 0",
    font=("Arial", 60, "bold"),
    text_color="#00ff99"
)

result.pack(pady=20)

# PREDICT FUNCTION

def predict_price():

    try:

        area = float(entries["property_area_sqft"].get())
        bed = int(entries["bedrooms"].get())
        bath = int(entries["bathrooms"].get())
        floor = int(entries["floors"].get())
        age = float(entries["property_age"].get())
        dist = float(entries["distance_to_city_km"].get())
        neigh = float(entries["neighborhood_quality_score"].get())
        quality = float(entries["construction_quality_rating"].get())

        luxury_score = (
            quality +
            neigh +
            8
        ) / 3

        villa = 1 if ptype.get() == "Villa" else 0

        user_df = pd.DataFrame([[

            area,
            bed,
            bath,
            floor,
            age,
            dist,
            neigh,
            quality,
            luxury_score,
            villa

        ]], columns=features)

        scaled = scaler.transform(user_df)

        prediction = model.predict(scaled)

        predicted_price = round(prediction[0], 2)

        # ANIMATION EFFECT
        result.configure(
            text="Calculating...",
            text_color="yellow"
        )

        app.update()

        result.configure(
            text=f"₹ {predicted_price}",
            text_color="#00ff99"
        )

        # SAVE DATABASE
        cursor.execute("""
        INSERT INTO predictions(predicted_price)
        VALUES(?)
        """, (predicted_price,))

        conn.commit()

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )

# PREDICT BUTTON

predict_btn = ctk.CTkButton(
    main,
    text="Predict House Price",
    width=420,
    height=70,
    font=("Arial", 24, "bold"),
    corner_radius=20,
    fg_color="#00bfff",
    hover_color="#0099cc",
    text_color="black",
    border_width=3,
    border_color="white",
    command=predict_price
)

predict_btn.pack(pady=20)

# FOOTER

footer = ctk.CTkLabel(
    main,
    text="Powered by Artificial Intelligence & Machine Learning",
    font=("Arial", 15),
    text_color="gray"
)

footer.pack(pady=30)

# RUN APP

app.mainloop()
