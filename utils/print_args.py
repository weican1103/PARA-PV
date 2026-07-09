def print_args(args):
    """Print experiment arguments in a simple format"""
    print('Args in experiment:')
    
    # Print args in a safer way to avoid encoding issues
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()
