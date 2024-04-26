from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
# standaralize
def std_l(df1):# need from sklearn.preprocessing import StandardScaler
    df = df1.copy()
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df.T).T # 处理股票数据，所有用这个来改变标准化方向
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    df1 = normalized_df
    #df1 = df1.mask((df1>3),3)
    #df1 = df1.mask((df1<-3),-3)
    return df1

# 定义截取函数，用来将各个数据表格进行掐头去尾的截取
def cutDF(data_df,date1,date2):
    date1 = pd.Timestamp(date1)
    date2 = pd.Timestamp(date2)
    data_df.index = pd.to_datetime(data_df.index)
    new_df = data_df[(data_df.index>=date1) & (data_df.index<=date2)]
    return new_df

# 事件数据的横纵处理
def getEventCtoR(rtndf, rawdf, rowname, colname, valuename, method="count", limitvalue=0):
    if (method == "count"):
        event = rawdf.groupby([rowname, colname])[valuename].count()
        event_df = event.unstack()
        event_df = event_df.mask(event_df <= limitvalue, np.nan)
        event_df = event_df.mask(event_df > limitvalue, 1)
        event_df = event_df.reindex(
            index=pd.date_range(
                rtndf.index[0], rtndf.index[-1]
            ),
            columns=rtndf.columns
        )
        event_df = event_df.fillna(0)
    if (method == "sum"):
        event = rawdf.groupby([rowname, colname])[valuename].sum()
        event_df = event.unstack()
        # event_df = event_df.mask(event_df<=limitvalue,np.nan)
        # event_df = event_df.mask(event_df>limitvalue,1)
        event_df = event_df.reindex(
            index=pd.date_range(
                rtndf.index[0], rtndf.index[-1]
            ),
            columns=rtndf.columns
        )
        event_df = event_df.fillna(0)

    return event_df

# 填充股票代码。
def fill_stkcd(arr):
    arr = pd.Series(list(arr))
    arr = arr.astype(str).str.zfill(6)
    return arr.tolist()

# 财务数据取最新值函数
def fetch_dayvalue(fac_date, data_df, value_name, rep_year="2022", rep_type='Q1', fic_per='3', ind1="TICKER_SYMBOL",
                   ind2="PUBLISH_DATE"):
    # fac_date表明因子生成时的日期，如"2024-03-01"，data_df指代原有的数据帧，它最好是已经整理过的，有股票代码，publish_date,REPORT_YEAR,REPORT_TYPE,FISCAL_PERIOD，且按序排列的)
    df = data_df[
        (data_df.REPORT_TYPE == rep_type) & (data_df.FISCAL_PERIOD == fic_per) & (data_df.REPORT_YEAR == rep_year)]

    # 获取delta一年的数据，只需回看470天，保证数据的时效性
    df = df[((df[ind2] - np.datetime64(fac_date)) <= np.timedelta64(0, 'D')) & (
                (np.datetime64(fac_date) - df[ind2]) < np.timedelta64(470, 'D'))]

    # 获取最新数据，即与fac_date时间相隔最近的数据
    def find_near(df):
        return df.loc[(np.abs(df.PUBLISH_DATE - np.datetime64(fac_date))).idxmin()]

    df = df.groupby('TICKER_SYMBOL').apply(find_near)

    return (df[value_name].astype(float))