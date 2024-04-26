# import 包
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def kesighfill(df1, df2, ffill_lim):
    DF1 = df1.copy()
    i = 5
    k = 5
    DF2 = df2.copy()
    DF1 = DF1.reindex(index=DF2.index)
    DF_test = DF1.iloc[5:, :]
    while ((((DF_test * DF2.iloc[k:, :]).count(axis=1) <= (DF2.iloc[k:, :].count(axis=1) * 0.5))).any()):
        DF1 = DF1.fillna(method='ffill', limit=i)
        k += 5
        DF_test = DF1.iloc[k:, :]

        if (k >= ffill_lim): print(f"填充达到指定上限:{ffill_lim}");break
    print("ffill reach:", k / 5 * 7)
    return DF1, min(ffill_lim, k / 5 * 7)


def max_drb(rt):
    high = 1
    db = 0
    for i in range(len(rt)):
        if (rt[i] > high):
            high = rt[i]
        if (((rt[i] - high) / high) < db):
            db = (rt[i] - high) / high
    return db


def single_fac_backtest(notfill_fac_df, next_rtndf, factor_name='', ffill_limit=240):
    df1 = notfill_fac_df.copy()
    df2 = next_rtndf.copy()
    df1.replace(0, np.nan, inplace=True)
    df2.replace(0, np.nan, inplace=True)

    # 第一张图要画数据填充程度，填充使用到函数kesighfill
    df1, fillnum = kesighfill(df1, df2, ffill_limit)

    df = df1.copy()
    df.reindex(index=df2.index)
    row_valid_ratio = (df * df2).notnull().sum(axis=1) / df2.notnull().sum(axis=1)

    # 第二张图要计算Ic，ICIR
    df1.index = pd.to_datetime(df1.index)
    df11 = df1.resample('M').first()
    df22 = df2.resample('M').sum()
    rank_df = df11.apply(lambda row: row.rank(), axis=1)
    IC = rank_df.apply(lambda row1: row1.corr(
        df22.loc[row1.name]) if row1.count() >= 100 and row1.name in df22.index else np.nan, axis=1)
    IC_mean = IC.mean()
    IC = IC.fillna(0)

    winrate = (IC[IC > 0].count()) / IC.count()

    # 第三张图要计算分组收益率，折线图

    pos_df = df1.rank(axis=1, pct=True) - 0.5
    pnl = {}
    for i in range(10):
        cond = (pos_df > (1 / 10 * (i) - 0.5)) & (pos_df <= (1 / 10 * (i + 1) - 0.5))
        tempdf = pos_df[cond]
        tempdf = tempdf.mask(tempdf < 100000000, 1)
        pnl[i] = ((tempdf.div(tempdf.sum(axis=1), axis=0) * df2).sum(axis=1))

    # 第四张图要计算分组超额，折线图
    pnl2 = {}
    for i in range(10):
        pnl2[i] = (pnl[i] + 1) / (1 + df2.mean(axis=1)) - 1

        # 第五张图要计算分组均值，折线图
    pnl_mean = pnl[0]
    for i in range(1, 10):
        pnl_mean += pnl[i]
    pnl_mean *= 0.1
    # 第七张图要画多、空、多空的收益，折线图,标题加上最大回撤好了，要用到回撤函数max_drb
    pos_df[pos_df > 0] = pos_df[pos_df > 0].div(pos_df[pos_df > 0].sum(axis=1), axis=0)
    pos_df[pos_df < 0] = -pos_df[pos_df < 0].div(pos_df[pos_df < 0].sum(axis=1), axis=0)
    pnl_long = ((pos_df[pos_df > 0] * df2).sum(axis=1) + 1) / (1 + df2.mean(axis=1))
    pnl_short = (1 + (pos_df[pos_df < 0] * df2).sum(axis=1)) * (1 + df2.mean(axis=1))
    pnl_ls = (pnl_long * pnl_short)

    db1 = max_drb(pnl_long)
    db2 = max_drb(pnl_short)
    db3 = max_drb(pnl_ls)

    # 下面是画图部分

    plt.figure(figsize=(54, 30))
    ## 第一张图
    # ax1 = plt.subplot2grid((4,2),(0,0),colspan = 1)
    # ax1.plot(row_valid_ratio, color='b', linestyle='-')
    # plt.xlabel('行数')
    # plt.ylabel('有效数据占比')
    # plt.title(f'填充后因子有效数据占比（/收益率有效数据） （目前已填充:{fillnum}天）')
    # plt.grid(True)
    ##plt.legend()

    # 第二张图
    ax2 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    ax2.plot(IC.cumsum(), color='tab:blue', label='Line Plot')
    ax2.set_ylabel('', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax3 = ax2.twinx()
    ax3.bar(IC.index, IC, color='tab:orange', alpha=0.7, label='Bar Plot', width=10)
    ax3.set_ylabel('', color='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:orange')

    ax2.tick_params(axis='y', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    plt.title('月度IC以及累计IC\nIC_mean:{:.4f}\nIC胜率{}'.format(IC_mean, winrate))
    plt.legend()

    # 　第三张图
    ax4 = plt.subplot2grid((2, 3), (0, 1), colspan=1)
    ax4.set_xlabel("time")
    ax4.set_ylabel("return")
    ax4.set_title("分十组累计收益")
    for i in range(10): ((1 + pnl[i]).cumprod() - 1).plot(label=f'{i} 组', ax=ax4)
    plt.legend()

    # 第四张图
    ax5 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
    ax5.set_xlabel("time")
    ax5.set_ylabel("return")
    ax5.set_title("分十组累计超额收益（benchmark：全市场）")
    for i in range(10): ((1 + pnl2[i]).cumprod() - 1).plot(label=f'{i} 组', ax=ax5)
    plt.legend()

    # 第五张图
    ax6 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    ax6.set_xlabel("time")
    ax6.set_ylabel("return")
    ax6.set_title("分十组累计收益（减均值）")
    for i in range(10): (((1 + pnl[i]) / (1 + pnl_mean)).cumprod() - 1).plot(label=f'{i} 组', ax=ax6)
    plt.legend()

    # 第六张图
    ax7 = plt.subplot2grid((2, 3), (1, 1), colspan=1)
    ax7.set_xlabel("time")
    ax7.set_ylabel("return")
    ax7.set_title("分十组超额收益月度均值（ben：全市场）")
    for i in range(10): ax7.bar(f'{i}', ((1 + pnl2[i]).prod()) ** (20 / len(pnl2[i])) - 1, label=f'{i} 组')
    plt.legend()

    # 第七张图
    ax8 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
    ax8.set_xlabel("time")
    ax8.set_ylabel("return")
    ax8.set_title("多空组合收益")
    ((pnl_long).cumprod() - 1).plot(label=f'long_MaxDB:{db1}', ax=ax8)
    ((pnl_short).cumprod() - 1).plot(label=f'short_MaxDB:{db2}', ax=ax8)
    ((pnl_ls).cumprod() - 1).plot(label=f'long-short_MaxDB:{db3}', ax=ax8)
    plt.legend()


def event_stats(Event_df,next_ret_df, event_name='', before_days=60,
                after_days=60):
    event_df = Event_df.replace(0, np.nan)
    alpha_ret_df = next_ret_df.sub(next_ret_df.mean(axis=1), axis=0)
    alpha_ret_nextday_df = alpha_ret_df

    alpha_ret_mean = pd.Series(index=range(-1 * before_days, after_days + 1), dtype='float32')

    for day in range(-1 * before_days, after_days + 1):
        ret_df = (event_df * alpha_ret_nextday_df.shift(-1 * day))
        ret_ls = ret_df.stack()

        alpha_ret_mean[day] = ret_df.mean(axis=1).mean()

    alpha_ret_cumsum = alpha_ret_mean.cumsum()
    alpha_ret_cumsum -= alpha_ret_cumsum[0]
    alpha_ret_cumsum = abs(alpha_ret_cumsum)

    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(311)
    ax1.bar(alpha_ret_mean.index, alpha_ret_mean, color='y', label='alpha mean daily')
    plt.legend()
    ax1.axvline(0, color='black')

    if len(alpha_ret_mean) >= 20:
        plt.xticks(alpha_ret_mean.index[::len(alpha_ret_mean) // 20])
    else:
        plt.xticks(alpha_ret_mean.index)
    ax1.title.set_text(
        f'alpha of event;mean:{round(alpha_ret_mean.loc[1:].mean() * 100, 4)}%'
    )
    ##############################
    ax2 = fig.add_subplot(312)
    ax2.bar(alpha_ret_cumsum.index, alpha_ret_cumsum, color='r', label='alpha cumsum')
    plt.legend()
    ax2.axvline(0, color='black')

    if len(alpha_ret_cumsum) >= 20:
        plt.xticks(alpha_ret_cumsum.index[::len(alpha_ret_cumsum) // 20])
    else:
        plt.xticks(alpha_ret_cumsum.index)
    ax2.title.set_text(
        f'alpha sum of event:{round(alpha_ret_mean.loc[1:].sum() * 100, 4)}%'
    )

    ###########################################################

    ax4 = fig.add_subplot(313)
    ax4.plot(event_df.index, event_df.count(axis=1))
    ax4.title.set_text(f'event share nums:{round(event_df.count(axis=1).mean(), 1)}')
    plt.legend()
    plt.grid()

    plt.suptitle(event_name)
    plt.plot()
    plt.show()