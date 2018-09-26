''' script for testing purposes '''


def times(x_1, y_2):
    ''' times together '''
    return x_1*y_2


def test_pass():
    ''' designed to pass '''
    assert times(1, 2) == 2

# leave this function commented out for CI to pass
# only uncomment to test making the CI fail
# def test_fail():
#       ''' designed to fail '''
#       assert times(1, 2) == 3


def main():
    ''' run fucntions '''
    test_pass()


# test coverage
if __name__ == '__main__':
    main()
