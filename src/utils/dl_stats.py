from typing import List
from torch.utils.data import DataLoader

import pandas as pd

def aggregate_batch_label_counts(
    dl:DataLoader, 
    label_list:List[str], 
    label_column_name:str='labels', 
    length_column_name:str='length'
):
    """Get DataLoader and Label_list and iterate dataloader to count label by batch.
    
    * This function is for making decision whether use bucketing or not.
    """
    df_list = []
    for i, batch in enumerate(dl):
        ll_df = pd.DataFrame({'labels': batch[label_column_name], 'length': batch[length_column_name]})
        ll_df['labels'] = ll_df['labels'].apply(lambda x:label_list[x])
        cnts = ll_df.labels.value_counts()
        ll_df = pd.DataFrame(data=cnts.values.reshape(1,-1), columns=cnts.index.tolist()).rename(index={0:f'batch-{i}'})
        df_list.append(ll_df)
    df = pd.concat(df_list)
    return df
    