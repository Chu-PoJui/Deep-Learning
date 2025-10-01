import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure outputs folder exists
outputs_dir = "outputs"
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# Read the Excel file from outputs folder
excel_path = os.path.join(outputs_dir, "model_performence.xlsx")
df = pd.read_excel(excel_path)
epoch = df['Epoch']

# Classify columns by name: Loss and Dice (treated as Accuracy)
loss_cols = [col for col in df.columns if "Loss" in col and col != "Epoch"]
dice_cols = [col for col in df.columns if "Dice" in col]

def calc_stats(series):
    """Calculate minimum, maximum, starting value, ending value, and percentage change"""
    start = series.iloc[0]
    end = series.iloc[-1]
    mn = series.min()
    mx = series.max()
    delta = (end - start) / start * 100 if start != 0 else np.nan
    return mn, mx, start, end, delta

def format_delta(delta):
    """Display change with ↑ (positive) and ↓ (negative), keeping two decimals with percentage sign"""
    if np.isnan(delta):
        return "NaN"
    elif delta > 0:
        return "↑ {:.2f}%".format(delta)
    elif delta < 0:
        return "↓ {:.2f}%".format(abs(delta))
    else:
        return "0.00"

# ==================== Loss Plot ====================
fig1, ax1 = plt.subplots(figsize=(10, 6))
stats_loss = []
colors_loss = []  # Record colors for each line

for col in loss_cols:
    # Plot line chart (without markers) and set line width to 1.25
    line, = ax1.plot(epoch, df[col], lw=1.25)
    colors_loss.append(line.get_color())
    mn, mx, start, end, delta = calc_stats(df[col])
    # Prefix the model name with a dash (you can change to other symbol if preferred)
    stats_loss.append(["— " + col, mn, mx, start, end, delta])

ax1.set_title("Training and Validation Loss vs. Epochs")
ax1.set_xlabel("Epochs")
ax1.xaxis.set_label_coords(0.5, -0.1)
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.axvline(x=200, color='black', linestyle='--', lw=1.5)

# Format the delta values using format_delta function
for row in stats_loss:
    row[5] = format_delta(row[5])

loss_table = pd.DataFrame(stats_loss, columns=["Model", "Min", "Max", "Start", "End", "Change (%)"])

# Set colWidths: the first column wider, and others as needed
col_widths = [0.25, 0.12, 0.12, 0.12, 0.12, 0.12]

table1 = ax1.table(cellText=loss_table.values,
                   colLabels=loss_table.columns,
                   cellLoc='left',
                   loc='bottom',
                   bbox=[-0.03, -0.5, 1, 0.3],
                   colWidths=col_widths)

# Remove all table cell borders and set text alignment to left
for key, cell in table1.get_celld().items():
    cell.set_linewidth(0)
    cell.get_text().set_ha("left")
    # For header row (row index 0), set bold and larger font
    if key[0] == 0:
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(12)

# Adjust the cell height to increase row spacing
for cell in table1.get_celld().values():
    cell.set_height(cell.get_height() * 1.5)

table1.auto_set_font_size(False)
table1.set_fontsize(10)

# Set the text color of the first column (Model) corresponding to each line's color
for i, color in enumerate(colors_loss):
    cell = table1[(i+1, 0)]  # Data starts from row 1 (row 0 is header)
    cell.get_text().set_color(color)

plt.subplots_adjust(bottom=0.45)
loss_image_path = os.path.join(outputs_dir, "loss_plot.png")
plt.savefig(loss_image_path)
print(f"Loss plot saved to {loss_image_path}")
plt.close(fig1)

# ==================== Accuracy (Dice) Plot ====================
fig2, ax2 = plt.subplots(figsize=(10, 6))
stats_dice = []
colors_dice = []

for col in dice_cols:
    line, = ax2.plot(epoch, df[col], lw=1.25)
    colors_dice.append(line.get_color())
    mn, mx, start, end, delta = calc_stats(df[col])
    stats_dice.append(["— " + col, mn, mx, start, end, delta])

ax2.set_title("Training and Validation Accuracy vs. Epochs")
ax2.set_xlabel("Epochs")
ax2.xaxis.set_label_coords(0.5, -0.1)
ax2.set_ylabel("Dice Score")
ax2.grid(True)
ax2.axvline(x=200, color='black', linestyle='--', lw=1.5)

for row in stats_dice:
    row[5] = format_delta(row[5])

dice_table = pd.DataFrame(stats_dice, columns=["Model", "Min", "Max", "Start", "End", "Change (%)"])

table2 = ax2.table(cellText=dice_table.values,
                   colLabels=dice_table.columns,
                   cellLoc='left',
                   loc='bottom',
                   bbox=[-0.03, -0.5, 1, 0.3],
                   colWidths=col_widths)

for key, cell in table2.get_celld().items():
    cell.set_linewidth(0)
    cell.get_text().set_ha("left")
    if key[0] == 0:
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(12)

for cell in table2.get_celld().values():
    cell.set_height(cell.get_height() * 1.5)

table2.auto_set_font_size(False)
table2.set_fontsize(10)

for i, color in enumerate(colors_dice):
    cell = table2[(i+1, 0)]
    cell.get_text().set_color(color)

plt.subplots_adjust(bottom=0.45)
dice_image_path = os.path.join(outputs_dir, "dice_plot.png")
plt.savefig(dice_image_path)
print(f"Dice plot saved to {dice_image_path}")
plt.close(fig2)
