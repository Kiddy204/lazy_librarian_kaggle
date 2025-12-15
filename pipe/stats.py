def print_clean_data_stats  (items_clean, interactions_clean, users_clean):
    print("="*50)
    print("Cleaning items")
    print("="*50)
    print(f"items_clean shape: {items_clean.shape}")
    print(f"items_clean columns: {items_clean.columns}")
    print(f"items_clean head:\n {items_clean.head()}")

    print("="*50)
    print("Cleaning interactions")
    print("="*50)

    print(f"interactions_clean shape: {interactions_clean.shape}")
    print(f"interactions_clean columns: {interactions_clean.columns}")
    print(f"interactions_clean head:\n {interactions_clean.head()}")


    print("="*50)
    print("Cleaning Users")
    print("="*50)
    print(f"users_clean shape: {len(users_clean)}")
    print(f"users_clean unique: {len(users_clean)}")
    print(f"max user id: {max(users_clean)}")
    print(f"min user id: {min(users_clean)}")
    print(f"users_clean head: {users_clean[:10]}")



def print_re_indexing_items_stats(interactions_clean, n_users, n_items):
    print("="*50)
    print("Re-indexing items")
    print("="*50)
    print(f"interactions_clean head: \n {interactions_clean.head()}")
    print(f"n_users: {n_users}")
    print(f"n_items: {n_items}")


def print_user_item_matrix_stats(train_user_item_matrix, test_user_item_matrix, val_user_item_matrix):
    print("="*50)
    print("Creating User Item Matrices")
    print("="*50)

    print("="*25)
    print("Train User Item Matrix")
    print("="*25)
    print(f"train_user_item_matrix shape: {train_user_item_matrix.shape}")
    print(f"train_user_item_matrix head:\n {train_user_item_matrix[:10]}")

    print("="*25)
    print("Test User Item Matrix")
    print("="*25)
    print(f"test_user_item_matrix shape: {test_user_item_matrix.shape}")
    print(f"test_user_item_matrix head:\n {test_user_item_matrix[:10]}")


    print("="*25)
    print("Val User Item Matrix")
    print("="*25)
    print(f"val_user_item_matrix shape: {val_user_item_matrix.shape}")
    print(f"val_user_item_matrix head:\n {val_user_item_matrix[:10]}")
