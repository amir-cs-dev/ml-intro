# ml_dark.py — Commander ML Dashboard (fixed watermark version)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Data ---
hours = [1, 2, 3, 4, 5, 6]
scores = [50, 55, 65, 70, 75, 80]
data = pd.DataFrame({"Hours Studied": hours, "Scores": scores})

# --- Model ---
X = [[h] for h in hours]
y = scores
model = LinearRegression()
model.fit(X, y)
hours_pred = 7
pred = model.predict([[hours_pred]])[0]

# --- Dark Mode Setup ---
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
fig.patch.set_facecolor("#0E0E0E")
fig.suptitle("Commander ML — Study Hours vs. Score", fontsize=14, fontweight="bold")

# --- Left Plot: Regression Line ---
axes[0].scatter(hours, scores, color="cyan", label="Actual Scores")
axes[0].plot(hours, model.predict(X), color="red", label="Regression Line")
axes[0].scatter(hours_pred, pred, color="lime", s=100, label=f"Prediction ({hours_pred} hrs)")
axes[0].text(hours_pred, pred + 2, f"{pred:.1f}", color="white", fontsize=9, ha="center")

axes[0].set_xlabel("Hours Studied")
axes[0].set_ylabel("Score")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.3)

# --- Right Plot: Data Table ---
axes[1].axis("off")
table = axes[1].table(
    cellText=data.values,
    colLabels=data.columns,
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.4)
for _, cell in table.get_celld().items():
    cell.set_facecolor("#1e1e1e")
    cell.set_edgecolor("#444")
    cell.get_text().set_color("white")

# --- Centered Watermark (visible in all viewers) ---
fig.text(
    0.5, 0.5,
    "© Commander Labs | github.com/amir-cs-dev",
    ha="center", va="center",
    fontsize=16,
    color="#00FFCC",
    alpha=0.15,
    weight="bold",
    rotation=30
)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
