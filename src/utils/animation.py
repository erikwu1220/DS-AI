import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def compare_animations(wds_, labels, total_time=96):
        """
        in:
        - wd, list of waterlevels to compare
        
        returns
        - animation"""

        def update_slider(val):
            for i in range(len(wds_)):
                skips = total_time // len(wds_[i])
                frame_index = val // (skips+1)
                imgs[i].set_array(wds_[i][frame_index])

                # redraw canvas while idle
                fig.canvas.draw_idle()

        
        height = len(wds_) // 3 + 1
        width = len(wds_) // height

        fig, axs = plt.subplots(height, width)
        axs = axs.ravel()

        imgs = []
        for i in range(len(wds_)):
            im = axs[i].imshow(wds_[i][0], cmap="Blues", origin='lower', animated=True)
            imgs.append(im)
            axs[i].set_title(f"{labels[i]}")

        # Add a slider
        ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Frame', 0, total_time, valinit=0, valstep=1)
        slider.on_changed(update_slider)

        plt.show()