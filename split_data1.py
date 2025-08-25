import pandas as pd

def split_dataframe(df, train_size=0.8, val_size=0.10):
    """
    Splits a time-ordered dataframe into train, validation, and test sets.
    Keeps time order → no leakage.
    """
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)

    print(f"Total rows: {n}")
    print(f"Train rows: {len(df_train)}")
    print(f"Validation rows: {len(df_val)}")
    print(f"Test rows: {len(df_test)}")

    return df_train, df_val, df_test