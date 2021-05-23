import os
import random
import lasio
import pickle
import pandas as pd
import numpy as np

FILES = [
    "data/15_9-F-1A.LAS.txt",
    "data/15_9-F-1B.LAS.txt",
    "data/15_9-F-1C.LAS.txt",
    "data/15_9-F-11A.LAS.txt",
    "data/15_9-F-11B.LAS.txt"
]
YES_FILES = [
    "data/15_9-F-1A.LAS.txt",
    "data/15_9-F-1B.LAS.txt",
    "data/15_9-F-11A.LAS.txt"
]

def read_filter_data(file_name=FILES[0], desired_columns=None):
    '''Returns the original df sliced according to where it seems to actually
    start collecting data. This is based on simply seeing where we start to see
    a consequtive set of data that is not null.
    '''
    df = lasio.read(file_name).df()
    possible_indices = df.index.tolist()
    possible_indices.sort(reverse=True)

    butt_index = -1
    last_null_value = -1
    stop_changing_count = 0
    for backwards in possible_indices:
        one_count = df.loc[df.index <= backwards].isnull().sum()["ABDCQF01"]
        if one_count == last_null_value:
            stop_changing_count += 1
        if stop_changing_count > 2:
            butt_index = backwards
            break
        last_null_value = one_count

    possible_indices.sort()

    front_index = -1
    for forwards in possible_indices:
        one_count = df.loc[(df.index >= forwards) & (df.index <= butt_index)].isnull().sum()
        one_count = one_count["ABDCQF01"] + one_count["ABDCQF03"]
        if one_count == 0:
            front_index = forwards
            break
    
    full_df = df.loc[(df.index >= front_index) & (df.index <= butt_index)]
    full_df = full_df[desired_columns]
    df_chunks = []
    starting_idx = 0
    for idx in range(len(full_df)):
        null_count = full_df.iloc[idx].isnull().sum().tolist()
        if null_count > 0:
            if starting_idx >= 0:
                print("slice", starting_idx, idx)
                this_slice = df.iloc[starting_idx:idx]
                df_chunks.append(this_slice)
                starting_idx = -1
            elif null_count == 1:
                print(full_df.iloc[idx].isnull().sum())
            print(null_count, starting_idx, idx)
        if starting_idx == -1 and null_count == 0:
            starting_idx = idx

    if starting_idx != -1:
        this_slice = df.iloc[starting_idx:]
        df_chunks.append(this_slice)


    return df_chunks




def train_valid_test_split(ratio="9:1:2", chunk_size=10, desired_columns=None):
    data = None
    if not os.path.isfile("cache/data.pkl"):
        df_list = []
        for yes_file in YES_FILES:
            print(desired_columns)
            dfs = read_filter_data(yes_file, desired_columns)
            for df in dfs:
                print(len(df))
            df_list = df_list + dfs
    
        
        full_df = pd.concat(df_list)
        df_min = full_df.min()
        df_max = full_df.max()
        
        
        # univ_columns = set(columns_list[0])
        # for columns in columns_list[1:]:
        #     univ_columns = univ_columns.intersection(set(columns))
        # univ_columns = list(univ_columns)
        univ_columns = desired_columns
        
        targets = []
        chunks = []
        for df in df_list:
            df = (df-df_min)/(df_max-df_min) # normalize
            for start_idx in range(len(df) - chunk_size - 1):
                one_chunk = df.iloc[start_idx:start_idx + chunk_size]
                if np.isnan(one_chunk.iloc[chunk_size//2 + 1]["DTS"]):
                    continue
                else:
                    targets.append(one_chunk.iloc[chunk_size//2 + 1]["DTS"])
                    del one_chunk["DTS"]
                    chunks.append(one_chunk.to_numpy())

        data = list(zip(chunks, targets))
        random.shuffle(data)
        pickle.dump(data, open("cache/data.pkl", "wb"))
    else:
        data = pickle.load(open("cache/data.pkl", "rb"))
    

    ratios = [int(i) for i in ratio.split(":")]
    to_valid = (len(data) * ratios[0]) // sum(ratios)
    to_test = to_valid + (len(data) * ratios[1]) // sum(ratios)


    train = data[:to_valid]
    valid = data[to_valid:to_test]
    test = data[to_test:]

    # with open("cache/train.pkl", "wb") as f:
    #     pickle.dump(train, f)
    # with open("cache/valid.pkl", "wb") as f:
    #     pickle.dump(valid, f)
    # with open("cache/test.pkl", "wb") as f:
    #     pickle.dump(test, f)
    # with open("cache/columns.txt", "w") as f:
    #     f.write("\n".join(univ_columns))
    
    # univ_columns.remove("DTS")
    return univ_columns, train, valid, test

if __name__ == "__main__":
    c,a,b,c = train_valid_test_split(desired_columns=["BS", "CALI", "DRHO", "DT", "DTS", "GR", "NPHI", "PEF", "RACEHM", "RACELM", "RHOB", "ROP", "RPCEHM", "RPCELM", "RT"])