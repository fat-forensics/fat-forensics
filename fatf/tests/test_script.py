def times(x, y):
    return x*y

def test_pass():
    assert times(1, 2) == 2

# def test_fail():
# 	assert times(1, 2) == 3

def main():
	test_pass()

#test coverage
if __name__ == '__main__':
	main()