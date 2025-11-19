import matplotlib.pyplot as plt
import numpy as np


def draw_polynomial_neuron(degree=22, x_sample=0.1, out_fname='neuron_diagram.png'):
    """Draw a schematic of a single linear neuron taking polynomial features x, x^2,...x^degree.

    - degree: number of polynomial features (e.g., 22)
    - x_sample: example input x used to show sample feature values (e.g., 0.1 -> 0.1,0.01,...)
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('off')

    # positions for input nodes
    left = 0.1
    neuron_x = 0.6
    neuron_y = 0.5

    ys = np.linspace(0.85, 0.15, degree)

    # draw input nodes
    for i, y in enumerate(ys, start=1):
        circ = plt.Circle((left, y), 0.02, fill=True, color='C0')
        ax.add_patch(circ)
        ax.text(left - 0.03, y, f'$x^{i}$', fontsize=9, ha='right', va='center')
        # sample value
        sample_val = x_sample ** i
        ax.text(left + 0.06, y, f'{sample_val:.3g}', fontsize=8, ha='left', va='center', color='gray')
        # arrow to neuron
        ax.annotate('', xy=(neuron_x - 0.05, neuron_y), xytext=(left + 0.02, y), arrowprops=dict(arrowstyle='->', lw=0.8))

    # neuron circle
    neuron = plt.Circle((neuron_x, neuron_y), 0.06, fill=True, color='C1', alpha=0.9)
    ax.add_patch(neuron)
    ax.text(neuron_x, neuron_y, 'Neuron', fontsize=10, ha='center', va='center', color='white')

    # weights and bias annotation
    ax.text(neuron_x + 0.12, neuron_y + 0.06, r'$y = \sum_{i=1}^{%d} w_i x^i + b$' % degree, fontsize=11)
    ax.text(neuron_x + 0.12, neuron_y - 0.06, r'weights: $w_1, w_2, \dots, w_{%d}$; bias $b$' % degree, fontsize=9, color='gray')

    # output arrow
    ax.annotate('', xy=(0.9, neuron_y), xytext=(neuron_x + 0.06, neuron_y), arrowprops=dict(arrowstyle='->', lw=1.2))
    ax.text(0.92, neuron_y, '$\hat{y}$', fontsize=12, va='center')

    # title and description
    ax.set_title('Single linear neuron with polynomial inputs (x, x^2, ..., x^{%d})' % degree, fontsize=12)
    ax.text(0.5, 0.02, f'Sample input: x = {x_sample} → first features = {x_sample:.3g}, {x_sample**2:.3g}, ...', ha='center', fontsize=9, color='gray')

    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
    plt.savefig(out_fname, dpi=150)
    print(f'Saved neuron diagram to {out_fname}')


if __name__ == '__main__':
    draw_polynomial_neuron(degree=22, x_sample=0.1, out_fname='neuron_diagram.png')
