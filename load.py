import lasio
import pickle
import pandas as pd
FILES = [
    "data/15_9-F-1A.LAS.txt",
    "data/15_9-F-1B.LAS.txt",
    "data/15_9-F-1C.LAS.txt",
    "data/15_9-F-11A.LAS.txt",
    "data/15_9-F-11B.LAS.txt"
]

def read_filter_data(file_name=FILES[0]):
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
    last_null_value = -1
    stop_changing_count = 0
    for forwards in possible_indices[10000:]:
        one_count = df.loc[(df.index >= forwards) & (df.index <= butt_index)].isnull().sum()["ABDCQF01"]
        if one_count == last_null_value:
            stop_changing_count += 1
        if stop_changing_count > 2:
            front_index = forwards
            break
        last_null_value = one_count
    
    return df.loc[(df.index >= front_index) & (df.index <= butt_index)]

def save_and_basic_analyze():
    columns = []
    f = open("cache/meta.txt", "w")
    for file_name in FILES:
        sliced_df = read_filter_data(file_name)
        columns.append((file_name, str(sliced_df.columns), str(len(sliced_df))))
        p_f = open(f'{file_name.replace(".txt", "").replace("data/", "cache/")}.pkl', "wb")
        pickle.dump(sliced_df, p_f)
        p_f.close()

    for col in columns:
        f.write(f"{col[0]}\n")
        f.write(f"{col[1]}\n")
        f.write(f"{col[2]}\n")
        f.write("\n")
    f.close()

def load_full():
    no_dts = []
    yes_dts = []
    for file_name in FILES:
        f = open(f'{file_name.replace(".txt", "").replace("data/", "cache/")}.pkl', "rb")
        df = pickle.load(f)
        f.close()
        if "DTS" in df.columns:
            print("YES", file_name)
            yes_dts.append(df)
        else:
            print("NO", file_name)
            no_dts.append(df)
    yes_concat = pd.concat(yes_dts)
    no_concat = pd.concat(no_dts)

    f = open("cache/yes_dts.pkl", "wb")
    pickle.dump(yes_concat, f)
    f.close()

    f = open("cache/no_dts.pkl", "wb")
    pickle.dump(no_concat, f)
    f.close()

de

            
if __name__ == "__main__":
    # save_and_basic_analyze()
    load_full()