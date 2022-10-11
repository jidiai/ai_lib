from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


class Monitor:
    '''
    TODO(jh): wandb etc
    TODO(jh): more functionality.
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.expr_log_dir)

    def get_expr_log_dir(self):
        return self.cfg.expr_log_dir

    def add_scalar(self, tag, scalar_value, global_step, *args, **kwargs):
        self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def add_multiple_scalars(self, main_tag, tag_scalar_dict, global_step, *args, **kwargs):
        for tag, scalar_value in tag_scalar_dict.items():
            tag = main_tag + tag
            self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step, *args, **kwargs):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, *args, **kwargs)

    def add_array(self, main_tag, image_array, xpid, ypid, global_step, color, *args, **kwargs):
        array_to_rgb(self.writer, main_tag, image_array, xpid, ypid, global_step, color, **kwargs)

    def close(self):
        self.writer.close()


def array_to_rgb(writer, tag, array, xpid, ypid, steps, color='bwr', **kwargs):
    matrix = np.array(array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = plt.cm.get_cmap(color)
    cax = ax.matshow(matrix, cmap=color_map)
    ax.set_xticklabels([''] + xpid, rotation=90)
    ax.set_yticklabels([''] + ypid)

    if kwargs.get('show_text', False):
        for (j, i), label in np.ndenumerate(array):
            ax.text(i, j, f'{label:.1f}', ha='center', va='center')

    fig.colorbar(cax)
    ax.grid(False)
    plt.tight_layout()

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    img = img.transpose(2, 0, 1)

    writer.add_image(tag, img, steps)
    plt.close(fig)





