import pymysql
import pandas as pd


cfghost = "183.2.215.126"
cfgport = 3306
cfguser = "abmuser"


# 定义连接函数和断开函数
def connect_toABM(cfgpassword):
    global conn
    conn = pymysql.connect(
        host=cfghost,
        port=cfgport,
        user=cfguser,
        password=cfgpassword,
    )
    global connectflag
    connectflag = 1


def closeconnect():
    conn.close()
    global connectflag
    connectflag = 0


# 定义获取纵表数据函数
def getDTFbySQL(connectflag, SQL, col):
    if (connectflag == 0):
        print("no connection");return 0
    else:
        cursor = conn.cursor()

        query = SQL
        cursor.execute(query)
        result = cursor.fetchall()
        df = pd.DataFrame().from_records(result, columns=col)
        cursor.close()
        return df