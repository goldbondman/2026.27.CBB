import pandas as pd

SOURCES={
 'spread':'https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/canonical/spreads/fg/spreads_fg_all.csv',
 'total':'https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/canonical/totals/fg/totals_fg_all.csv',
}
for name,url in SOURCES.items():
    df=pd.read_csv(url,nrows=5)
    print('\n###',name)
    print('columns=',list(df.columns))
    print(df.head(3).to_string())
