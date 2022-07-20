import numpy as np
import matplotlib.pyplot as plt


def plot_result(guide_img, source_img, prediction, target_img=None, data_type="rgb", fig_size=(16, 4)):
    cmap = "Spectral"

    if len(guide_img.shape) > 2:
        guide_img = np.rollaxis(guide_img, 0, 3)

        if data_type == "sat":
            guide_img = (guide_img[:, :, [2, 1, 0]])

        elif data_type == "rgb":
            guide_img = (guide_img[:, :, [0, 1, 2]])

        else:
            guide_img = np.mean(guide_img, axis=2)

    guide_min = np.percentile(guide_img, 0.05, axis=(0, 1), keepdims=True)  
    guide_max = np.percentile(guide_img, 99.95, axis=(0, 1), keepdims=True)  
    guide_img = (guide_img - guide_min) / (guide_max - guide_min)
    guide_img = np.clip(guide_img, 0, 1)

    if target_img is not None:
        vmin = np.min(target_img)
        vmax = np.max(target_img)

        f, axarr = plt.subplots(1, 4, figsize=fig_size)

        if len(guide_img.shape) > 2:
            axarr[0].imshow(guide_img)
        else:
            axarr[0].imshow(guide_img, cmap="gray")
            
        axarr[1].imshow(source_img, vmin=vmin, vmax=vmax, cmap=cmap)
        axarr[2].imshow(target_img, vmin=vmin, vmax=vmax, cmap=cmap)
        axarr[3].imshow(prediction, vmin=vmin, vmax=vmax, cmap=cmap)
        
        titles = ['Guide', 'Source', 'Target',
                  'Prediction (RMSE {:.3f})'.format(np.sqrt(np.mean((target_img - prediction) ** 2)))]

    else:
        vmin = np.min(source_img)
        vmax = np.max(source_img)
        
        f, axarr = plt.subplots(1, 3, figsize=fig_size)
    
        if len(guide_img.shape) > 2:
            axarr[0].imshow(guide_img)
        else:
            axarr[0].imshow(guide_img, cmap="gray")

        axarr[1].imshow(source_img, vmin=vmin, vmax=vmax, cmap=cmap)
        axarr[2].imshow(prediction, vmin=vmin, vmax=vmax, cmap=cmap)

        titles = ['Guide', 'Source', 'Prediction'] 

    for i, ax in enumerate(axarr):
        ax.set_axis_off()
        ax.set_title(titles[i])

    plt.tight_layout()
    return f, axarr
