import numpy as np
from scipy.special import kv, kvp

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')

import argparse
import os
import lambda_k_plotting as plot
reload(plot)
import analytic_condition as ac
from analytic_condition import conditions
import equil_solver as es


parser = argparse.ArgumentParser(description='make paper plots')
parser.add_argument('high_epsilon_path',
                    help="path of high epsilon profile stability results")
parser.add_argument('mid_epsilon_path',
                    help="path of mid epsilon profile stability results")
parser.add_argument('low_epsilon_path',
                    help="path of low epsilon profile stability results")
args = parser.parse_args()
high_epsilon_path = os.path.join(args.high_epsilon_path,
                                 'meshes.npz')
mid_epsilon_path = os.path.join(args.mid_epsilon_path,
                                'meshes.npz')
low_epsilon_path = os.path.join(args.low_epsilon_path,
                                'meshes.npz')


## Figure 1 ##
##############

fig, axes = plt.subplots(2, 2,
                         figsize=(6.69, 6.69),
                         sharex=False,
                         sharey=False)

example_ax = axes[0][0]
kink_ax = axes[0][1]
sausage_ax = axes[1][0]
delta_ax = axes[1][1]

## Example Plot ##
##################

lambda_bar = np.linspace(0, 4.25, 500)
k_bar = np.linspace(0, 1.5, 500)
mesh = np.meshgrid(lambda_bar, k_bar)
lambda_bar_mesh, k_bar_mesh = mesh[0], mesh[1]


d_w_sausage = conditions(k_bar_mesh,
                         lambda_bar_mesh,
                         epsilon=0.1,
                         m=0,
                         delta=0.1)
d_w_kink = conditions(k_bar_mesh,
                      lambda_bar_mesh,
                      epsilon=0.1,
                      m=1,
                      delta=0.1)

stability_sausage = d_w_sausage < 0
stability_kink = d_w_kink < 0
stability = stability_kink.astype(float)
stability[stability_sausage] = 2


cmap = colors.ListedColormap(['white',
                              'lightgrey',
                              'lightgrey'])

example_ax.contourf(lambda_bar_mesh, k_bar_mesh, stability, cmap=cmap,
                    levels=[0., 0.5, 1.5, 2.], hatches=[None, '/', 'x'])
contour_lines = example_ax.contour(lambda_bar_mesh, k_bar_mesh, stability,
                                   levels=[0., 1.5, 2.], colors='black', linewidths=2)

example_ax.clabel(contour_lines,
                  manual=([1.5, 1.1],
                          [2.5, 0.75]),
                  fmt={0.: ' Kink Boundary ', 1.5: ' Sausage Boundary '}, fontsize=10)

kruskal_shafranov = lambda_bar_mesh > 2 * k_bar_mesh

contour_lines = example_ax.contour(lambda_bar_mesh, k_bar_mesh, kruskal_shafranov,
                                   levels=[0.5], colors='black', linewidths=2)
contour_lines.collections[0].set_linestyle('--')
example_ax.clabel(contour_lines,
                  manual=([[1.8, 0.95]]),
                  fmt={0.5: 'Kruskal-Shafranov'}, fontsize=10)

#example_ax.plot([0, 3.], [0., 1.5], '--', c='black', lw=5)

example_ax.set_xlabel(r'$\bar{\lambda}$')
#example_ax.xaxis.labelpad = 20
plt.setp(example_ax.get_xticklabels())
example_ax.set_xticks(np.arange(0., 5, 1.))

example_ax.set_ylabel(r'$\bar{k}$', rotation='horizontal')
plt.setp(example_ax.get_yticklabels())
example_ax.set_yticks(np.arange(0., 2.0, 0.5))

example_ax.plot(2*np.sqrt(2), 0.0, 'o', lw=2, color='black')
example_ax.annotate('Taylor', color='black',
                    xy=(2*np.sqrt(2), 0),
                    xytext=(2*np.sqrt(2) + 0.025, 0.25),
                    rotation=45, fontsize=10)

sns.despine()


## Kink Plot ##
###############

delta = 0.0
stability = np.zeros(lambda_bar_mesh.shape) - 0.2
for i, epsilon in enumerate(np.arange(0.1, 1.9, 0.3)):

    d_w_kink = conditions(k_bar_mesh,
                          lambda_bar_mesh,
                          epsilon=epsilon,
                          m=1,
                          delta=delta)

    stability_epsilon = d_w_kink < 0
    stability[stability_epsilon] = epsilon

levels = np.arange(0.1, 1.9, 0.3)
contour = kink_ax.contourf(lambda_bar_mesh,
                           k_bar_mesh,
                           stability,
                           levels=levels)
contour.cmap.set_over(sns.xkcd_rgb['red brown'])
contour.cmap.set_under('white')
contour.set_clim(-0.1, 0.)
colors_list = ['grey', 'white', 'white', 'white', 'white']
contour_lines = kink_ax.contour(lambda_bar_mesh,
                               k_bar_mesh,
                               stability,
                               levels=levels,
                               colors=colors_list,
                               linewidths=2)
kink_ax.clabel(contour_lines,
                              manual=([1.6, 1.3],
                                      [2.1, 1.2],
                                      [2.3, 1.1],
                                      [2.4, 0.8],
                                      [2.4, 0.5]),
                               fmt=r'$ \epsilon = %.1f $', colors=colors_list,
                               fontsize=12)
for line in contour_lines.collections:
    line.set_linestyle('solid')

line_x = np.linspace(0, 3, 50)
line_y = np.linspace(0, 1.5, 50)

line_y_masked = np.ma.masked_inside(line_y, 1.05, 1.18)

kink_ax.plot(line_x, line_y_masked, '--', c='black', lw=2)


kink_ax.plot([0, 3.], [0., 1.5], '--', c='black', lw=2)
kink_ax.set_xlabel(r'$\bar{\lambda}$')
#kink_ax.xaxis.labelpad = 20
plt.setp(kink_ax.get_xticklabels())
kink_ax.set_xticks(np.arange(0., 5, 1.))

kink_ax.set_ylabel(r'$\bar{k}$', rotation='horizontal')
plt.setp(kink_ax.get_yticklabels())
kink_ax.set_yticks(np.arange(0., 2.0, 0.5))

sns.despine()

## Sausage Plot ##
##################

delta = -0.7
stability = np.zeros(lambda_bar_mesh.shape) - 0.2
for i, epsilon in enumerate(np.arange(0, 1.2, 0.2)):

    d_w_sausage = conditions(k_bar_mesh,
                            lambda_bar_mesh,
                            epsilon=epsilon,
                            m=0,
                            delta=delta)

    stability_epsilon = d_w_sausage < 0
    stability[stability_epsilon] = epsilon

levels = np.arange(0., 1.2, 0.2)
contour = sausage_ax.contourf(lambda_bar_mesh,
                              k_bar_mesh,
                              stability,
                              levels=levels)
contour.cmap.set_over('darkgreen')
contour.cmap.set_under('white')
contour.set_clim(-0.2, -0.1)

colors_list = ['grey', 'white', 'white', 'white']

contour_lines = sausage_ax.contour(lambda_bar_mesh,
                                   k_bar_mesh,
                                   stability,
                                   levels=levels,
                                   colors=colors_list,
                                   linewidths=2)
for line in contour_lines.collections:
    line.set_linestyle('solid')


sausage_ax.clabel(contour_lines,
                              manual=([2.3, 1.3],
                                      [1.75, 0.55],
                                      [2.0, 0.4],
                                      [2.1, 0.2]),
                               fmt=r'$ \epsilon = %.1f$', fontsize=12)

sausage_ax.plot([0, 3.], [0., 1.5], '--', c='black', lw=2)
sausage_ax.set_xlabel(r'$\bar{\lambda}$')
#sausage_ax.xaxis.labelpad = 20
plt.setp(sausage_ax.get_xticklabels())
sausage_ax.set_xticks(np.arange(0., 5, 1.))

sausage_ax.set_ylabel(r'$\bar{k}$', rotation='horizontal')
plt.setp(sausage_ax.get_yticklabels())
sausage_ax.set_yticks(np.arange(0., 2.0, 0.5))
sns.despine()


## Delta dependence ##
######################

epsilon = 0.2
stability = np.zeros(lambda_bar_mesh.shape) - 1.2
for i, delta in enumerate(np.arange(-1.0, 1.6, 0.5)):

    d_w_delta = conditions(k_bar_mesh,
                           lambda_bar_mesh,
                           epsilon=epsilon,
                           m=0,
                           delta=delta)

    stability_delta = d_w_delta < 0
    stability[stability_delta] = delta

levels = np.arange(-1.0, 1.6, 0.5)
contour = delta_ax.contourf(lambda_bar_mesh,
                            k_bar_mesh,
                            stability,
                            levels=levels,
                            extend='both')
levels = np.arange(-1.0, 1.6, 0.5)

colors_list = ['grey', 'white', 'white', 'white', 'white']

contour_lines = delta_ax.contour(lambda_bar_mesh,
                                 k_bar_mesh,
                                 stability,
                                 levels=levels,
                                 colors=colors_list,
                                 linewidths=2)
contour.cmap.set_over('darkgreen')
contour.cmap.set_under('white')
contour.set_clim(-1.3, -1.0)

for line in contour_lines.collections:
    line.set_linestyle('solid')

sausage_ax.clabel(contour_lines,
                  manual=([2., 0.85],
                          [2.4, 0.75],
                          [2.8, 0.65],
                          [3.0, 0.55],
                          [3.3, 0.45]),
                  fmt=r' $ \delta = \mathbf{%.1f}$ ',
                  colors=colors_list, fontsize=12)

delta_ax.plot([0, 3.], [0., 1.5], '--', c='black', lw=2)
delta_ax.set_xlabel(r'$\bar{\lambda}$')
#delta_ax.xaxis.labelpad = 20
plt.setp(delta_ax.get_xticklabels())
delta_ax.set_xticks(np.arange(0., 5, 1.))

delta_ax.set_ylabel(r'$\bar{k}$', rotation='horizontal')
plt.setp(delta_ax.get_yticklabels())
delta_ax.set_yticks(np.arange(0., 2.0, 0.5))

plt.tight_layout()

plt.figtext(0.001, 0.97, '(a)')
plt.figtext(0.001, 0.5, '(c)')
plt.figtext(0.49, 0.97, '(b)')
plt.figtext(0.49, 0.5, '(d)')

fig.savefig('../figures/figure1.eps', dpi=300)


## Fig 2 Profiles ##
####################
def normalized_single_plot(profile, axes, ylim, letter, styles=['-', '--', '-.'],
                           legend_loc=None):
    r"""
    """
    splines = profile.get_splines()
    beta_spl = profile.beta(profile.r)
    r1 = np.linspace(0, 1., 250)

    j_skin = profile.j_skin
    b_z0 = profile.b_z0


    axes.yaxis.tick_left()
    axes.xaxis.tick_bottom()

    j_z_line = axes.plot(r1, splines['j_z'](r1)/j_skin, lw=2.5, label=r'$\bar{j}_z$', ls=styles[0])
    b_theta_line = axes.plot(r1, splines['b_theta'](r1), lw=2.5, label=r'$\bar{B}_\theta$', ls=styles[1])
    p_line = axes.plot(r1, splines['pressure'](r1), lw=2.5, label=r'$\bar{p}$', ls=styles[2])
    #j_z_line = axes.plot(r1, splines['j_z'](r1)/j_skin, lw=2.5, ls=styles[0])
    #b_theta_line = axes.plot(r1, splines['b_theta'](r1), lw=2.5, ls=styles[1])
    #p_line = axes.plot(r1, splines['pressure'](r1), lw=2.5, ls=styles[2])
    axes.set_xlim(0, 1.)

    axes.set_ylim(0., ylim)

    if legend_loc:
        lgd = axes.legend(bbox_to_anchor=legend_loc,
                          loc='upper center',
                          borderaxespad=0.25,
                          ncol=3,
                          frameon=True,
                          handlelength=1.8,
                          columnspacing=1.1)
    else:
        lgd = None

    #axes.annotate('core', xy=(0.27, ylim*4.5/7.))
    #axes.annotate('transition', xy=(0.624, ylim*4.5/7.))
    #axes.annotate('skin', xy=(0.772, ylim*4.5/7.))
    #axes.annotate('transition', xy=(0.85, ylim*4.5/7.))

    axes.set_xlabel(r'$r$')
    axes.axvline(x=0.6, color='grey', ls='--', lw=2.5, alpha=0.5)
    axes.axvline(x=0.775, color='grey', ls='--', lw=2.5, alpha=0.5)
    axes.axvline(x=0.825, color='grey', ls='--', lw=2.5, alpha=0.5)
    axes.yaxis.set_ticks(np.arange(0.0, 1.8, 0.7))

    plt.setp(axes.get_xticklabels())
    plt.setp(axes.get_yticklabels())
    return lgd

fig = plt.figure(figsize=(3.75,5.5))
axes1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
axes2 = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=1)
axes3 = plt.subplot2grid((3, 2), (2, 0), colspan=2, rowspan=1)
fig.subplots_adjust(wspace=0.05)
fig.subplots_adjust(hspace=0.5)

profile = es.UnitlessSmoothedCoreSkin(k_bar=1, lambda_bar=1, epsilon=0.7,
                                      core_radius_norm=0.6,
                                      transition_width_norm=0.175,
                                      skin_width_norm=0.05)

lgd = normalized_single_plot(profile, axes1, 1.8, 'a', legend_loc=[0.51, 1.8])

profile = es.UnitlessSmoothedCoreSkin(k_bar=1, lambda_bar=1, epsilon=0.5,
                                      core_radius_norm=0.6,
                                      transition_width_norm=0.175,
                                      skin_width_norm=0.05)

normalized_single_plot(profile, axes2, 1.8, 'b')

profile = es.UnitlessSmoothedCoreSkin(k_bar=1, lambda_bar=1, epsilon=0.1,
                                      core_radius_norm=0.6,
                                      transition_width_norm=0.175,
                                      skin_width_norm=0.05)

normalized_single_plot(profile, axes3, 1.8, 'c')
#axes1.legend(loc='best')
#axes2.legend(loc='best')
#axes3.legend(loc='best')
#plt.figlegend([l1, l2, l3],
#           [r'$\bar{j}_z$',
#            r'$\bar{B}_\theta$',
#            r'$\bar{p}$'], 'upper left',
#           bbox_to_anchor = (0.5, 0.5))

#plt.tight_layout()
plt.figtext(-0.0, 0.90, '(a)')
plt.figtext(-0.0, 0.60, '(b)')
plt.figtext(-0.0, 0.30, '(c)')
#plt.show()
fig.subplots_adjust(hspace=0.75)
sns.despine()
plt.savefig('../figures/figure2.eps', dpi=300,
            bbox_extra_artists=(lgd,), bbox_inches='tight')

## Fig 3 Numerical Stability Space ##
#####################################

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.colors import SymLogNorm, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter, FixedFormatter
import matplotlib.patches as patches
import matplotlib.ticker as ticker

def plot_lambda_k_space_dw(axes, filename, epsilon, name, mode_to_plot='m_neg_1',
                           show_points=False, lim=None, levels=None, log=True,
                           linthresh=1E-7, bounds=(1.5, 3.0), norm=True,
                           analytic_compare=False,
                           label_pos=((0.5, 0.4), (2.1, 0.4), (2.8, 0.2)),
                           delta_values=[-1,0,1],
                           interpolate=False,
                           cmap=None, hatch=False,
                           figsize=None,
                           save_as=None,
                           return_ax=False,
                           hatch_sausage_gap=False):

    epsilon_case = np.load(filename)
    lambda_a_mesh = epsilon_case['lambda_a_mesh']
    k_a_mesh = epsilon_case['k_a_mesh']
    external_m_neg_1 = epsilon_case['d_w_m_neg_1']
    external_sausage = epsilon_case['d_w_m_0']
    epsilon_case.close()

    if hatch_sausage_gap:
        external_sausage_gap = np.where((lambda_a_mesh > 3.) & (external_sausage > 0))
        external_sausage[external_sausage_gap] = np.nan

    instability_map = {'m_0': external_sausage,
                       'm_neg_1': external_m_neg_1}


    kink_pal = sns.blend_palette([sns.xkcd_rgb["dandelion"],
                                  sns.xkcd_rgb["white"]], 7, as_cmap=True)
    kink_pal = sns.diverging_palette(73, 182, s=72, l=85, sep=1, n=9, as_cmap=True)
    sausage_pal = sns.blend_palette(['orange', 'white'], 7, as_cmap=True)
    sausage_pal = sns.diverging_palette(49, 181, s=99, l=78, sep=1, n=9, as_cmap=True)

    if cmap:
        instability_palette = {'m_0': cmap,
                               'm_neg_1': cmap}
    else:
        instability_palette = {'m_0': sausage_pal,
                               'm_neg_1': kink_pal}




    if interpolate:
        instability_map['m_neg_1'] = interpolate_nans(lambda_a_mesh,
                                                      k_a_mesh,
                                                      instability_map['m_neg_1']
                                                      )

    values = instability_map[mode_to_plot]

    if norm:
        values = values / np.nanmax(np.abs(values))
    else:
        values = values

    if levels:
        if log:
            plot = axes.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                levels=levels, norm=SymLogNorm(linthresh))
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(plot, cax=cax, label=r'$\delta W$')
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.set_label(label=r'$\delta W$', size=10, rotation=0, labelpad=1)
            contourlines = axes.contour(lambda_a_mesh, k_a_mesh,
                                      values, levels=levels,
                                      colors='grey',
                                      norm=SymLogNorm(linthresh), linewidths=0.8)

        else:
            norm = BoundaryNorm(levels, 256)
            plot = axes.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                levels=levels, norm=norm)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(plot, cax=cax, label=r'$\delta W$')
            cbar.set_label(label=r'$\delta W$', size=10, rotation=0, labelpad=1)
            contourlines = axes.contour(lambda_a_mesh, k_a_mesh,
                                       values, levels=levels,
                                       colors='grey', linewidths=0.8)
    else:
        if log:
            plot = axes.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot],
                                norm=SymLogNorm(linthresh))
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(plot, cax=cax, label=r'$\delta W$')
            cbar.set_label(label=r'$\delta W$', size=10, rotation=0, labelpad=1)
            contourlines = axes.contour(lambda_a_mesh, k_a_mesh,
                                       values, colors='grey',
                                       norm=SymLogNorm(linthresh), linewidths=0.8)
        else:
            plot = plt.contourf(lambda_a_mesh, k_a_mesh, values,
                                cmap=instability_palette[mode_to_plot])
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(plot, cax=cax, label=r'$\delta W$')
            cbar.set_label(label=r'$\delta W$', size=10, rotation=0, labelpad=1)
            contourlines = axes.contour(lambda_a_mesh, k_a_mesh,
                                       values, colors='grey', linewidths=0.8)

    if lim:
        plot.set_clim(lim)

    cbar.add_lines(contourlines)

    cbar_lines = cbar.lines[0]
    number_of_lines = len(cbar_lines.get_linewidths())

    linewidths = []
    linestyles = []
    cbar_zero_line = 4
    for line in xrange(number_of_lines):
        if line < cbar_zero_line:
            linewidths.append(0.8)
            linestyles.append('--')
        elif line == cbar_zero_line:
            linewidths.append(0.8)
            linestyles.append('-')
        else:
            linewidths.append(0.8)
            linestyles.append('-')

    cbar_lines.set_linewidths(linewidths)
    cbar_lines.set_linestyles(linestyles)

    axes.plot([0.01, 0.1, 1.0, 2.0, 3.0],
             [0.005, 0.05, 0.5, 1.0, 1.5], color='black', lw=1.5)

    axes.set_axis_bgcolor(sns.xkcd_rgb['white'])

    lambda_bar_analytic = np.linspace(0.01, 4., 750)
    k_bar_analytic = np.linspace(0.01, 1.5, 750)
    (lambda_bar_mesh_analytic,
     k_bar_mesh_analytic) = np.meshgrid(lambda_bar_analytic, k_bar_analytic)

    if analytic_compare:
        analytic_comparison(mode_to_plot, k_bar_mesh_analytic,
                            lambda_bar_mesh_analytic, epsilon, label_pos)

    if show_points:
        axes.scatter(lambda_a_mesh, k_a_mesh, marker='o', c='b', s=2)

    axes.set_ylim(0.01, bounds[0])
    axes.set_xlim(0.01, bounds[1])
    axes.set_xticks(np.arange(0., 4.5, 1.0))
    axes.set_yticks(np.arange(0., 2.0, 0.5))
    plt.setp(axes.get_xticklabels(), fontsize=10)
    plt.setp(axes.get_yticklabels(), fontsize=10)
    axes.set_ylabel(r'$\bar{k}$', rotation='horizontal', fontsize=10)
    axes.set_xlabel(r'$\bar{\lambda}$', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    def my_formatter_fun(x):
        if x == 0:
            return r'$0$'
        if np.sign(x) > 0:
            return r'$10^{%i}$' % np.int(np.log10(x))
        else:
            return r'$-10^{%i}$' % np.int(np.log10(np.abs(x)))
    labels = [my_formatter_fun(level) for level in levels]
    cbar.ax.set_yticklabels(labels)
    sns.despine(ax=axes)

    if hatch:
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()
        xy = (xmin,ymin+0.1)
        width = xmax - xmin
        height = ymax - ymin
        p = patches.Rectangle(xy, width, height, hatch='X'*2,
                              zorder=-10, edgecolor='#FEFFE0',
                              facecolor='#6C2605')
        axes.add_patch(p)
    if hatch_sausage_gap:
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()
        xmin = 3.
        xy = (xmin, ymin+0.01)
        width = xmax - xmin
        height = ymax - ymin
        p = patches.Rectangle(xy, width, height, hatch='X'*2,
                              zorder=-10, edgecolor='#FEFFE0',
                              facecolor='#005B31')
        axes.add_patch(p)
    cbar.ax.yaxis.set_ticks_position('right')


fig = plt.figure(figsize=(6.69, 6.69))

kink_ax1 = plt.subplot2grid((9, 6), (0, 0), colspan=3, rowspan=3)
sausage_ax1 = plt.subplot2grid((9, 6), (0, 3), colspan=3, rowspan=3)
kink_ax2 = plt.subplot2grid((9, 6), (3, 0), colspan=3, rowspan=3)
sausage_ax2 = plt.subplot2grid((9, 6), (3, 3), colspan=3, rowspan=3)
kink_ax3 = plt.subplot2grid((9, 6), (6, 0), colspan=3, rowspan=3)
sausage_ax3 = plt.subplot2grid((9, 6), (6, 3), colspan=3, rowspan=3)


## High epsilon
plot_lambda_k_space_dw(kink_ax1, high_epsilon_path,
                       1., 'ep12-m1', mode_to_plot='m_neg_1',
                       levels=[-1, -1e-1, -1e-2, -1e-3,
                               0, 1e-3, 1e-2, 1e-1, 1],
                       norm=True, analytic_compare=False,
                       log=True,
                       label_pos=None,
                       interpolate=False, cmap="YlOrBr_r",
                       hatch=True,
                       bounds=(1.5, 4.3))


plot_lambda_k_space_dw(sausage_ax1, high_epsilon_path,
                       1., 'ep12-m1', mode_to_plot='m_0',
                       levels=[-1, -1e-1, -1e-2, -1e-3,
                               0, 1e-3, 1e-2, 1e-1, 1],
                       norm=True, analytic_compare=False,
                       log=True,
                       label_pos=None,
                       interpolate=False, cmap="YlGn_r",
                       bounds=(1.5, 4.3),
                       hatch_sausage_gap=True)



## Mid epsilon
plot_lambda_k_space_dw(kink_ax2, mid_epsilon_path,
                       1., 'ep12-m1', mode_to_plot='m_neg_1',
                       levels=[-1, -1e-1, -1e-2, -1e-3,
                               0, 1e-3, 1e-2, 1e-1, 1],
                       norm=True, analytic_compare=False,
                       log=True,
                       label_pos=None,
                       interpolate=False, cmap="YlOrBr_r",
                       hatch=True,
                       bounds=(1.5, 4.3))


plot_lambda_k_space_dw(sausage_ax2, mid_epsilon_path,
                       1., 'ep12-m1', mode_to_plot='m_0',
                       levels=[-1, -1e-1, -1e-2, -1e-3,
                               0, 1e-3, 1e-2, 1e-1, 1],
                       norm=True, analytic_compare=False,
                       log=True,
                       label_pos=None,
                       interpolate=False, cmap="YlGn_r",
                       hatch_sausage_gap=True,
                       bounds=(1.5, 4.3))




## Low epsilon
plot_lambda_k_space_dw(kink_ax3, low_epsilon_path,
                       1., 'ep12-m1', mode_to_plot='m_neg_1',
                       levels=[-1, -1e-1, -1e-2, -1e-3,
                               0, 1e-3, 1e-2, 1e-1, 1],
                       norm=True, analytic_compare=False,
                       log=True,
                       label_pos=None,
                       interpolate=False, cmap="YlOrBr_r",
                       bounds=(1.5, 4.3),
                       hatch=True)

plot_lambda_k_space_dw(sausage_ax3, low_epsilon_path,
                       1., 'ep12-m1', mode_to_plot='m_0',
                       levels=[-1, -1e-1, -1e-2, -1e-3,
                                    0, 1e-3, 1e-2, 1e-1, 1],
                       norm=True, analytic_compare=False,
                       log=True,
                       label_pos=None,
                       bounds=(1.5, 4.3),
                       interpolate=False, cmap="YlGn_r",
                       hatch=True)



plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.5)

plt.figtext(0.01, 0.98, '(a)', fontsize=10)
plt.figtext(0.01, 0.655, '(b)', fontsize=10)
plt.figtext(0.01, 0.33, '(c)', fontsize=10)
plt.figtext(0.49, 0.98, '(d)', fontsize=10)
plt.figtext(0.49, 0.655, '(e)', fontsize=10)
plt.figtext(0.49, 0.33, '(f)', fontsize=10)

plt.savefig('../figures/figure3.eps', dpi=300)

### Ratio Plots ###
###################

def sausage_kink_ratio(axes, filename, xy_limits=None, cmap=None, save_as=None,
                       levels=None, zero_line=2, label_lines=False, cbar_zero_line=2):
    r"""
    Plot ratio of sausage and kink potential energies.
    """
    meshes = np.load(filename)
    lambda_bar_mesh = meshes['lambda_a_mesh']
    k_bar_mesh = meshes['k_a_mesh']
    external_m_neg_1 = meshes['d_w_m_neg_1']
    external_sausage = meshes['d_w_m_0']
    meshes.close()

    sausage_stable_region = np.invert((external_sausage < 0))
    ratio = np.abs(external_sausage / external_m_neg_1)
    ratio[sausage_stable_region] = np.nan
    ratio_log = np.log10(ratio)

    if not cmap:
        cmap = sns.light_palette(sns.xkcd_rgb['red orange'],
                                 as_cmap=True)
    if levels:
        contours = axes.contourf(lambda_bar_mesh, k_bar_mesh,
                                ratio_log, cmap=cmap, levels=levels)
    else:
        contours = axes.contourf(lambda_bar_mesh, k_bar_mesh,
                                ratio_log, cmap=cmap)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = plt.colorbar(contours, cax=cax, format=FormatStrFormatter(r'$10^{%i}$'))
    colorbar.set_label(r'$\frac{\delta W_{m=0}}{\delta W_{m=1}}$', rotation=0,
                       labelpad=10,
                       fontsize=10)
    if levels:
        lines = axes.contour(lambda_bar_mesh, k_bar_mesh,
                            ratio_log, colors='black', levels=levels, linewidths=1)
    else:
        lines = axes.contour(lambda_bar_mesh, k_bar_mesh,
                            ratio_log, colors='black', linewidths=1)
    plt.setp(lines.collections[zero_line], linewidth=2)
    colorbar.add_lines(lines)

    cbar_lines = colorbar.lines[0]
    number_of_lines = len(cbar_lines.get_linewidths())

    linewidths = []
    linestyles = []
    for line in xrange(number_of_lines):
        if line < cbar_zero_line:
            linewidths.append(1)
            linestyles.append('--')
        elif line == cbar_zero_line:
            linewidths.append(2)
            linestyles.append('-')
        else:
            linewidths.append(1)
            linestyles.append('-')

    cbar_lines.set_linewidths(linewidths)
    cbar_lines.set_linestyles(linestyles)

    axes.plot([0, 3.], [0., 1.5], '--', c='black', lw=2)
    axes.set_xlabel(r'$\bar{\lambda}$', fontsize=10)
    plt.setp(axes.get_xticklabels(), fontsize=10)
    axes.set_xticks(np.arange(0., 4.5, 1.0))

    axes.set_ylabel(r'$\bar{k}$', rotation='horizontal', fontsize=10)
    plt.setp(axes.get_yticklabels(), fontsize=10)
    axes.set_yticks(np.arange(0., 2.0, 0.5))

    if label_lines:
        plt.clabel(lines)

    if xy_limits:
        axes.set_ylim((xy_limits[0], xy_limits[1]))
        axes.set_xlim((xy_limits[2], xy_limits[3]))
    plt.setp(axes.get_xticklabels(), fontsize=10)
    plt.setp(axes.get_yticklabels(), fontsize=10)
    axes.set_xlim((0, 4.3))
    sns.despine(ax=axes)
    colorbar.ax.yaxis.set_ticks_position('right')
    colorbar.ax.tick_params(labelsize=8)

fig = plt.figure(figsize=(3.75,6.69))

ratio_ax1 = plt.subplot2grid((9, 3), (0, 0), colspan=3, rowspan=3)
ratio_ax2 = plt.subplot2grid((9, 3), (3, 0), colspan=3, rowspan=3)
ratio_ax3 = plt.subplot2grid((9, 3), (6, 0), colspan=3, rowspan=3)

sausage_kink_ratio(ratio_ax1, high_epsilon_path,
                   cmap="Greys",
                   zero_line=2, label_lines=False,
                   cbar_zero_line=2)

sausage_kink_ratio(ratio_ax2, mid_epsilon_path,
                   cmap="Greys",
                   zero_line=1, label_lines=False,
                   cbar_zero_line=1)

sausage_kink_ratio(ratio_ax3, low_epsilon_path,
                   cmap="Greys",
                   levels=[-4, -2, 0, 2, 4, 6, 15],
                   zero_line=2, label_lines=False,
                   cbar_zero_line=2)
plt.tight_layout()

plt.figtext(0.05, 0.96, '(a)', fontsize=10)
plt.figtext(0.05, 0.64, '(b)', fontsize=10)
plt.figtext(0.05, 0.32, '(c)', fontsize=10)
plt.savefig('../figures/figure4.eps', dpi=300)
