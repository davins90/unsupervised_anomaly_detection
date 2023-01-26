import pandas as pd
import gdown

def data_retrieval(url):
    """
    
    """
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    print(dwn_url)
    if '1pCcXTt5OURpAoo0JCuDqFrQt3vo_xmT3' in url:
        df = pd.read_csv(dwn_url)
    else:
        output = "../../../data_lake/input/transaction_raw.csv"
        gdown.download(dwn_url,output,quiet=False)
        # remove from dataframe columns related to Vesta's generated features
        df = pd.read_csv(output,usecols=lambda x: "V" not in x)
    return df