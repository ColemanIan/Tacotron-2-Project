# Adjusting the diagram layout to move diagonally (top-left to bottom-right)
fig, ax = plt.subplots(figsize=(12, 9))
ax.axis('off')

# Define diagonal box layout
boxes_diagonal = {
    'Text Input': (0.1, 0.85, 0.14, 0.06),
    'Embedding\n(148→512)': (0.25, 0.75, 0.14, 0.06),
    'Encoder Conv\n(3×Conv1D,512)': (0.4, 0.65, 0.16, 0.06),
    'Encoder BiLSTM\n(512→256×2)': (0.55, 0.55, 0.16, 0.06),
    'Attention Mechanism': (0.7, 0.45, 0.16, 0.06),
    'Attention RNN\n(768→1024)': (0.7, 0.35, 0.16, 0.06),
    'Decoder RNN\n(1536→1024)': (0.55, 0.25, 0.16, 0.06),
    'Linear Projection\n(1536→80)': (0.4, 0.15, 0.16, 0.06),
    'Gate Layer\n(1536→1)': (0.25, 0.05, 0.14, 0.06),
    'Mel Spectrogram\n(80-dim)': (0.55, 0.15, 0.16, 0.06),
    'Prenet\n(80→256→256)': (0.7, 0.25, 0.16, 0.06),
    'Postnet\n(5×Conv1D)': (0.85, 0.05, 0.14, 0.06),
    'Final Output': (0.85, -0.05, 0.14, 0.06)
}

# Draw diagonal boxes
for name, (x, y, w, h) in boxes_diagonal.items():
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", edgecolor='black', facecolor='whitesmoke')
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, name, ha='center', va='center', fontsize=9)

# Diagonal arrows for clarity
arrows_diagonal = [
    ('Text Input', 'Embedding\n(148→512)'),
    ('Embedding\n(148→512)', 'Encoder Conv\n(3×Conv1D,512)'),
    ('Encoder Conv\n(3×Conv1D,512)', 'Encoder BiLSTM\n(512→256×2)'),
    ('Encoder BiLSTM\n(512→256×2)', 'Attention Mechanism'),
    ('Attention Mechanism', 'Attention RNN\n(768→1024)'),
    ('Attention RNN\n(768→1024)', 'Decoder RNN\n(1536→1024)'),
    ('Decoder RNN\n(1536→1024)', 'Linear Projection\n(1536→80)'),
    ('Linear Projection\n(1536→80)', 'Gate Layer\n(1536→1)'),
    ('Linear Projection\n(1536→80)', 'Mel Spectrogram\n(80-dim)'),
    ('Mel Spectrogram\n(80-dim)', 'Prenet\n(80→256→256)'),
    ('Prenet\n(80→256→256)', 'Attention RNN\n(768→1024)'),
    ('Mel Spectrogram\n(80-dim)', 'Postnet\n(5×Conv1D)'),
    ('Postnet\n(5×Conv1D)', 'Final Output')
]

# Draw arrows
for src, dst in arrows_diagonal:
    src_x, src_y, src_w, src_h = boxes_diagonal[src]
    dst_x, dst_y, dst_w, dst_h = boxes_diagonal[dst]
    ax.annotate("",
                xy=(dst_x + dst_w / 2, dst_y + dst_h),
                xytext=(src_x + src_w / 2, src_y),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.2))

plt.title("Tacotron 2 Information Flow Diagram (Diagonal Layout)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
