import pymysql

host_name = 'labuser58mysql.mysql.database.azure.com'
user_name = 'labuser58'
password = 'dlwn2157!'
database_name = 'classicmodels'

db = pymysql.connect(
    host=host_name,
    port=3306,
    user=user_name,
    passwd=password,
    db=database_name,
    charset='utf8',
    ssl={"fake_flag_to_enable_tls":True} #보안설정 바뀜 확인할것
    ) # db 연결

import pandas as pd

SQL = 'select * from employees'
df = pd.read_sql(SQL, db)

type(df)
print(df)

SQL = '''
create table tempTable(
  id int AUTO_INCREMENT,
  user_name varchar(10) NOT NULL,
  phone varchar(30) NULL,
  PRIMARY KEY(id)
)
'''

cursor = db.cursor()
cursor.execute(SQL)
cursor.close()

SQL = 'insert into tempTable(user_name,phone) values ("IU", "010-1111-1111")'

cursor = db.cursor()
cursor.execute(SQL)
cursor.close()

SQL = 'select * from temptable'
pd.read_sql(SQL, db)