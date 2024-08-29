
import datetime
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def plot_corr(fname, corr_tiled, corr_bigbox, title_tiled, title_bigbox, labels, vmin=-0.3, vmax=0.3):    
    nbin = corr_bigbox.shape[0] // len(labels)
    tick_positions = [nbin/2 + nbin * i for i in range(len(labels))]

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    cax = ax[0].imshow(corr_tiled, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax[0], shrink=0.6)
    ax[0].set_title(title_tiled, fontsize=10)

    cax = ax[1].imshow(corr_bigbox, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax[1], shrink=0.6)
    ax[1].set_title(title_bigbox, fontsize=10)

    cax = ax[2].imshow(corr_bigbox - corr_tiled, cmap='bwr', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=ax[2], shrink=0.6)
    ax[2].set_title("BigBox - Tiled Correlation", fontsize=10)

    for axes in ax.flatten():
        axes.set_xticks(tick_positions, labels, fontsize=8)
        axes.set_yticks(tick_positions, labels, fontsize=8, rotation=90, va='center')
        axes.invert_yaxis()

    fig.savefig(fname, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_stats(fname, title, title_tiled, title_bigbox, ell, nu, means_tiled, means_bigbox, labels, stds_tiled=None, stds_bigbox=None, ratio_range=[0.95, 1.05]):
    nbin = means_bigbox.shape[0] // len(labels)

    fig = plt.figure(figsize=(15, 6))

    gs_master = GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])
    gs_ell = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0, j], height_ratios=[3, 1], hspace=0.01) for j in range(2)]
    gs_nu = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[1, j], height_ratios=[3, 1], hspace=0.01) for j in range(3)]

    axes = [fig.add_subplot(gs_ell[i][0]) for i in range(2)] + [fig.add_subplot(gs_nu[i][0]) for i in range(3)]
    ratio_axes = [fig.add_subplot(gs_ell[i][1]) for i in range(2)] + [fig.add_subplot(gs_nu[i][1]) for i in range(3)]

    for i, label in enumerate(labels):
        ax = axes[i]
        ratio_ax = ratio_axes[i]
        
        if i < 2:
            tiled_data = means_tiled[i*nbin:i*nbin+nbin]
            bigbox_data = means_bigbox[i*nbin:i*nbin+nbin]
            if stds_tiled is not None:
                ax.errorbar(ell, tiled_data, yerr=stds_tiled[i*nbin:i*nbin+nbin], label="Tiled", capsize=2)
            else:
                ax.plot(ell, tiled_data, label="Tiled")

            if stds_bigbox is not None:
                ax.errorbar(ell, bigbox_data, yerr=stds_bigbox[i*nbin:i*nbin+nbin], label="BigBox", capsize=2, c="tab:orange")
            else:
                ax.plot(ell, bigbox_data, label="BigBox", c="tab:orange")
            
            ratio = bigbox_data / tiled_data
            ratio_ax.plot(ell, ratio, label="BigBox / Tiled", c="tab:purple")
            ratio_ax.set_xscale('log')
            ratio_ax.set_ylim(ratio_range)
            ratio_ax.set_xticks([300, 500, 1000, 2000, 3000])
            ratio_ax.set_xticklabels([300, 500, 1000, 2000, 3000])
            
            ax.set_title(label)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xticks([300, 500, 1000, 2000, 3000])
            ax.set_xticklabels([300, 500, 1000, 2000, 3000])
        else:
            tiled_data = means_tiled[i*nbin:i*nbin+nbin]
            bigbox_data = means_bigbox[i*nbin:i*nbin+nbin]
            if stds_tiled is not None:
                ax.errorbar(nu, tiled_data, yerr=stds_tiled[i*nbin:i*nbin+nbin], label="Tiled", capsize=2)
            else:
                ax.plot(nu, tiled_data, label="Tiled")

            if stds_bigbox is not None:
                ax.errorbar(nu, bigbox_data, yerr=stds_bigbox[i*nbin:i*nbin+nbin], label="BigBox", capsize=2, c="tab:orange")
            else:
                ax.plot(nu, bigbox_data, label="BigBox", c="tab:orange")
            
            ratio = bigbox_data / tiled_data
            ratio_ax.plot(nu, ratio, label="BigBox / Tiled", c="tab:purple")
            ratio_ax.set_ylim(ratio_range)
            
            ax.set_title(label)
            ax.set_yscale('log')
            
        ratio_ax.set_ylabel("Ratio")
        ratio_ax.axhline(1, color='gray', linestyle='--')  # Add a horizontal line at 1 for reference

        # Hide the x-ticks for the main plots to avoid redundancy
        plt.setp(ax.get_xticklabels(), visible=False)

    # Add text to the gs_master[0, 2] panel
    text_ax = fig.add_subplot(gs_master[0, 2])
    text_ax.text(0.5, 0.8, title, fontsize=16, ha='center', va='center')
    # text the datetime from system
    now = datetime.datetime.date(datetime.datetime.now())
    text_ax.text(0.5, 0.6, f'{now}', fontsize=12, ha='center', va='center')
    text_ax.text(0.5, 0.4, title_tiled, fontsize=12, ha='center', va='center')
    text_ax.text(0.5, 0.2, title_bigbox, fontsize=12, ha='center', va='center')
    text_ax.axis('off')  # Turn off the axis for the text panel

    # Automatically adjust spacing to avoid overlaps
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(fname, bbox_inches='tight')
    plt.show()
