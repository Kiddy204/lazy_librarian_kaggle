import pandas as pd
def create_train_test_split(interactions, test_size=0.2, val_size=0.2):
    train_rows = []
    test_rows = []
    val_rows = []

    for u, user_df in interactions.sort_values("datetime").groupby("u"):
        n = len(user_df)
        if n < 3:
            continue

        train_end = int((1 - test_size - val_size) * n)
        test_end  = int((1 - val_size) * n)

        train_rows.append(user_df.iloc[:train_end])
        test_rows.append(user_df.iloc[train_end:test_end])
        val_rows.append(user_df.iloc[test_end:])

    return (
        pd.concat(train_rows),
        pd.concat(test_rows),
        pd.concat(val_rows),
    )