from .run import run


# TODO: As an exercise to the reader, try implementing command-line arguments
# for the names of these files using the argparse module:
# https://docs.python.org/3/library/argparse.html
run(prices_filename='prices.png',
    returns_filename='returns.png')
