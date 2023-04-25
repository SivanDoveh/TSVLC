import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib
import random
import os
import json
import seaborn as sns

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

class Radar(object):
    def __init__(self, figure, title,limits,start_radar, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = [a if a <=360. else a - 360. for a in np.arange(90, 90+360, 360.0/self.n)]
        self.limits = limits
        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
        self.start_radar = start_radar
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=17)
        self.ax.tick_params(pad=20)
        # self.ax.grid(color='black')
        self.n_ticks = 4

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, limit in zip(self.axes, self.angles, self.limits):
            ticks = [(limit / self.n_ticks) * i for i in range(self.n_ticks + 1)]
            ax.set_rgrids(ticks, angle=angle, label=ticks)

            ax.spines['polar'].set_visible(False)
            ax.set_ylim(self.start_radar, limit)  # limit


    def plot(self, values,color, alpha,label,limits):
        limits = np.array(limits)
        # Hacky way to adjust for the first scale which apparently
        # is leading...

        values[1:] = (values[1:] / limits[1:]) * limits[0]
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values,"-", lw=2,color=color,alpha=alpha, label=label,)
        self.ax.legend(loc="right", bbox_to_anchor=(1, 1), )
        self.ax.fill(angle, values, color=color,alpha=alpha)


def zs_generate_chart(args,output_folder,corpus_path,task,chart_type,models,name,ep_eval,radar_legends,start_radar):
    """
    output_folder : str
        The path of folder that saves all evaluation results.
    corpus_path : str
        The path of json file that records corresponding file names of all metrics
    task : {'itm' | 'itc'}
        Task type of output result.
    chart_type : str
        Type of chart.
    models : list
        List of model name.
    """
    m = json.load(open('training/'+corpus_path))
    arrs = []
    model_data_radar=[]
    colors = [(1.0, 0.4980392156862745, 0.054901960784313725),(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)]
    if radar_legends:
        legends = radar_legends
    else:
        legends = models
    for model in models:
        per_data_radar = {
            'vg': [0] * m.__len__(),
            'vaw': [0] * m.__len__(),
            'swig': [0] * m.__len__(),
            'hake': [0] * m.__len__(),
        }
        filepath = os.path.join(output_folder,model)
        score_list = []
        ep = ep_eval
        for ind, item in enumerate(m.keys()):
            data_num = len(m[item].keys())
            data_score = []
            for data in m[item].keys():
                score = 0
                file_num = len(m[item][data])
                for file in m[item][data]:

                    json_name = os.path.join(filepath,f"{file}_{ep}.json")
                    if not os.path.exists(json_name):
                        print(f"{file}_{ep}.json has not been evaluated. model name: {model}")
                        return
                    else:
                        m1 = json.load(open(json_name))
                        score += m1["total_acc"]
                per_data_radar[data][ind] = score/file_num
                data_score.append(score/file_num)
            score_list.append(sum(data_score)/data_num)
        # for z in [5,1]:
        #     json_name = os.path.join(filepath, f"top{z}_zs_{ep}.json")
        #     m1 = json.load(open(json_name))
        #     score = m1["total_acc"]
        #     score_list.append(score)

        arrs.append(score_list)
        model_data_radar.append(per_data_radar)
        print(f'{model} {score_list}')
    print('')

    # fig_ = plt.figure(figsize=(8, 8))
    # data = [['A-Color', 'A-Material', 'A-Size',
    #             "A-State", "A-Action", "R-Action", "R-Spatial"],
    #             ('', (np.array(arrs)[:,6:]).tolist())]
    #
    # spoke_labels = data.pop(0)
    # title, case_data = data[0]
    # list_limits=[]
    # for c in range(len(spoke_labels)):
    #     c_v = [k[c] for k in case_data]
    #     list_limits.append(min(5 + (max(c_v)) * 100, 100))
    # limits = np.array(list_limits)
    # radar = Radar(fig_, spoke_labels,limits,start_radar)
    # case_data_100=[[ 100 * i for i in inner ] for inner in case_data]
    # for ii, model_data in enumerate(case_data_100):
    #     print(f'{models[ii]} {model_data}')
    # print('')
    # for ii,model_data in enumerate(case_data_100):
    #     c = colors[ii]
    #     radar.plot(model_data,color=c, alpha=0.3, label=legends[ii],limits=limits)
    #
    # for data_name in ['vg','swig','hake','vaw']:
    #     fig_ = plt.figure(figsize=(8, 8))
    #     test_names = ['O-Large', 'O-Medium', 'O-Small', 'O-Center', 'O-Mid', 'O-Margin', 'A-Color', 'A-Material', 'A-Size',
    #              "A-State", "A-Action", "R-action", "R-spatial",]
    #     tests_in_data=[]
    #     case_data=[]
    #     for i in range(len(models)):
    #         tests_in_data.append([x for ind, item in enumerate(model_data_radar[i][data_name]) if item != 0 for x in (test_names[ind],item)])
    #         case_data.append(tests_in_data[i][1::2])
    #
    #     spoke_labels = tests_in_data[0][::2]
    #
    #     list_range = []
    #     # for c in range(len(spoke_labels)):
    #     #     c_v = [k[c] for k in case_data]
    #     #     list_range.append([str(int(x)) for x in np.linspace(0, min((max(c_v)) * 100, 100), 4).tolist()])
    #
    #     # radar = Radar(fig_, spoke_labels, list_range)
    #     case_data_100 = [[100 * i for i in inner] for inner in case_data]
    #     print(data_name)
    #     print(spoke_labels)
    #     for ii, model_data in enumerate(case_data_100):
    #         # c = colors[ii]
    #         print(f'{models[ii]} {model_data}')
    #     print('')
    #
    #         # radar.plot(model_data, color=c, alpha=0.3, label=models[ii])
    # #     plt.legend(loc="best", bbox_to_anchor=(1, 1), )
    # #     plt.title((data_name+'_'+name).replace('_', ' '), loc='left')
    # #     plt.savefig(os.path.join(output_folder.split('eval_jsons')[0] + 'radars/', data_name+'_'+name + '.png'), bbox_inches='tight')
    # #
    # #
    # #     # ax = fig_.add_subplot(int(f'13{1}'), projection='radar')
    # #     #
    # #     # ax.set_rgrids(np.arange(0,1,0.1))
    # #     # lines = []
    # #     # labels = models
    # #     # for ii in range(len(models)):
    # #     #     d=tests_in_data[ii][1::2]
    # #     #     c = colors[ii]
    # #     #     line = ax.plot(theta, d, label=labels[ii], color=c)
    # #     #     lines.append(line)
    # #     #     ax.fill(theta, d, alpha=0.25, color=c)
    # #     # ax.set_varlabels(spoke_labels)
    # #     # plt.legend(loc="best", bbox_to_anchor=(1, 1), )
    # #     # plt.title(data_name+' '+name)
    # #     # plt.savefig(os.path.join(output_folder.split('eval_jsons')[0] + 'radars/', data_name+'_'+name + '.png'), bbox_inches='tight')
    #
    #
