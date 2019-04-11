from fatf.vis import lime as vislime
import numpy as np
import matplotlib.pyplot as plt 

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
    fig = vislime.plot_lime(data)
    axes = fig.axes
    values = list(data.values())
    keys = list(data.keys())
    order = [text.get_text() for text in axes[0].get_yticklabels()]
    assert order == [x[0] for x in data['class0']]
    assert len(axes) == len(keys)
    for i in range(len(axes)):
        assert axes[i].title.get_text() == keys[i]
        assert ({x.get_width() for x in axes[i].patches} == 
                {value[1] for value in values[i]})
