# File names for all tests must begin with 'test_' or end with '_test'
# functions inside these files should start with 'test_'
# pytest documentation can provide lots of examples with how to construct test scripts
import fatf

def test_should_fail():
     assert(0==1)

def test_should_pass():
     assert(0==0)