#!/usr/bin/env python3
#
# PROGRAMMER: Christiaan Lombard
# DATE CREATED: 2020-11-21
# REVISED DATE: 2020-11-21
# PURPOSE: Helper functions for common command line tasks
#
#

import argparse


def print_command_line_arguments(in_arg):
    """Print commandline arguments keys and values

    Args:
        in_arg (Namespace): Commandline arguments
    """

    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:")
        for key, value in vars(in_arg).items():
            print("%s = %s" % (key, value))


def str2bool(v):
    """Parse a boolean-like command line argument

    Args:
        v (str): A boolean-like string

    Raises:
        argparse.ArgumentTypeError: If not a boolean-like string

    Returns:
        bool: The parsed bool
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
