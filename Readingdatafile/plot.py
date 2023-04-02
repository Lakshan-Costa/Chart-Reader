import matplotlib.pyplot as plt

rectangles = [
    [(328.0, 545.0), (342.0, 545.0), (342.0, 554.0), (328.0, 554.0)],
    [(298.0, 545.0), (314.0, 545.0), (314.0, 556.0), (298.0, 556.0)],
    [(285.0, 545.0), (296.0, 545.0), (296.0, 553.0), (285.0, 553.0)],
    [(268.0, 543.0), (277.0, 543.0), (277.0, 554.0), (268.0, 554.0)],
]

fig, ax = plt.subplots()

for rectangle in rectangles:
    left, top, right, bottom = rectangle
    rect = plt.Rectangle((left, bottom), right - left, top - bottom, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

ax.set_xlim(left=0, right=600)
ax.set_ylim(bottom=0, top=600)
plt.show()
