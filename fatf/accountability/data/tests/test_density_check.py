import numpy as np
import pytest

from fatf.accountability.data.density_check import DensityCheck

testdata_struct = np.array([(74, 52), ( 3, 86), (26, 56), (70, 57), (48, 57), (30, 98),
       (41, 73), (24,  1), (44, 66), (62, 96), (63, 51), (26, 88),
       (94, 64), (59, 19), (14, 88), (16, 15), (94, 48), (41, 25),
       (36, 57), (37, 52), (21, 42)],
      dtype=[('Age', '<i4'), ('Weight', '<i4')])

testdata = np.array([(74, 52), ( 3, 86), (26, 56), (70, 57), (48, 57), (30, 98),
       (41, 73), (24,  1), (44, 66), (62, 96), (63, 51), (26, 88),
       (94, 64), (59, 19), (14, 88), (16, 15), (94, 48), (41, 25),
       (36, 57), (37, 52)])


@pytest.mark.parametrize("input_dataset",
                         [(testdata)])
def test_check_alpha(input_dataset):
    with pytest.raises(TypeError) as errmsg:
        mdl = DensityCheck(input_dataset, alpha = 'a')
    
    with pytest.raises(ValueError) as errmsg:
        mdl = DensityCheck(input_dataset, alpha = -1)
        
    with pytest.raises(ValueError) as errmsg:
        mdl = DensityCheck(input_dataset, alpha = 1.5)
        
@pytest.mark.parametrize("input_dataset",
                         [(testdata)])
def test_check_neighbours(input_dataset):
    with pytest.raises(TypeError) as errmsg:
        mdl = DensityCheck(input_dataset, neighbours = 'a')
    
    with pytest.raises(ValueError) as errmsg:
        mdl = DensityCheck(input_dataset, neighbours = -1)
        
    with pytest.raises(ValueError) as errmsg:
        mdl = DensityCheck(input_dataset, neighbours = 100)
    
expected_output_scores = np.array([1369., 2312., 1186., 1114.,  701., 1780.,  954., 4021.,  808.,
       2048.,  970., 1360., 2890., 1865., 1825., 3385., 3265., 1186.,
       1061., 1114.])  
@pytest.mark.parametrize("input_dataset, expected_output_scores",
                         [(testdata, expected_output_scores)])
def test_scores(input_dataset, expected_output_scores):
    mdl = DensityCheck(input_dataset, 
                       neighbours = 7,
                       alpha = 0.90)
    output_scores = mdl.get_scores()
    print(output_scores)
    assert np.all(output_scores == expected_output_scores)