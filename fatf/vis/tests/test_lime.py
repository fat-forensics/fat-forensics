from fatf.vis import lime as vislime
import numpy as np
import matplotlib.pyplot as plt 

from fatf.vis._utils import _get_bar_plot_data


def test_plot_lime():
    data = {
        'class0': [('feat0 <= 0.00', -0.4153762474280945), 
                   ('0.50 < feat1 <= 1.00', -0.28039957101809865), 
                   ('0.07 < feat2 <= 0.22', 0.03778942895340688), 
                   ('0.34 < feat3 <= 0.58', -0.007232109279325609)], 
        'class1': [('0.50 < feat1 <= 1.00', 0.2028506569431207), 
                   ('0.07 < feat2 <= 0.22', -0.07699173494077427), 
                   ('feat0 <= 0.00', 0.01986873036503522), 
                   ('0.34 < feat3 <= 0.58', -0.018218096708096074)], 
        'class2': [('feat0 <= 0.00', 0.39550751706305864), 
                   ('0.50 < feat1 <= 1.00', 0.07754891407497788), 
                   ('0.07 < feat2 <= 0.22', 0.039202305987367285), 
                   ('0.34 < feat3 <= 0.58', 0.02545020598742168)]}
    # All classes have same feature bounds so plot should sharey axis
    # and first axes will have correct labels and the rest will be empty
    bar_widths = [[value[1] for value in array] 
                  for array in list(data.values())]
    keys = list(data.keys())
    x_range = [-0.45, 0.43]
    y_range = [-0.09, 4.09]
    fig = vislime.plot_lime(data)
    axes = fig.axes
    y_labels = [[x[0] for x in data['class0']], [], []]
    assert len(axes) == len(keys)
    for i in range(len(axes)):
        p_title, p_x_ticks, p_x_range, p_y_ticks, p_y_range, p_bar_width = \
            _get_bar_plot_data(axes[i])
        assert y_labels[i] == p_y_ticks
        assert p_title == keys[i]
        assert set(p_bar_width) == set(bar_widths[i])
        assert np.isclose(x_range, p_x_range, atol=1e-2).all()
        assert np.isclose(y_range, p_y_range, atol=1e-2).all()

    # Test when sharey is False and the yticklabels are unique for each axis
    del data['class1'][2]
    bar_widths = [[value[1] for value in array] 
                  for array in list(data.values())]
    x_range = [-0.45, 0.43]
    y_range = [[-0.09, 4.09], [-0.04, 3.04], [-0.09, 4.09]]
    fig = vislime.plot_lime(data)
    axes = fig.axes
    y_labels = [[x[0] for x in data[i]] for i in keys]
    assert len(axes) == len(keys)
    for i in range(len(axes)):
        p_title, p_x_ticks, p_x_range, p_y_ticks, p_y_range, p_bar_width = \
            _get_bar_plot_data(axes[i])
        assert y_labels[i] == p_y_ticks
        assert p_title == keys[i]
        assert set(p_bar_width) == set(bar_widths[i])
        assert np.isclose(x_range, p_x_range, atol=1e-2).all()
        assert np.isclose(y_range[i], p_y_range, atol=1e-2).all()
