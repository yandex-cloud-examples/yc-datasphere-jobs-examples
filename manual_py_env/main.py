if __name__ == '__main__':
    # envzy explorer can't find imports outside global namespace
    import pandas as pd
    from data import arr

    with open('result.txt', 'w') as f:
        f.write(str(float(pd.Series(arr()).mean())))
