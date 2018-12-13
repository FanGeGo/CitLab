#coding:utf-8
'''
数据库操作
'''
import psycopg2

class dbop:

    def __init__(self,insert_index=0):

        self._insert_index=insert_index
        self._insert_values=[]
        logging.debug("connect database with normal cursor.")
        self._db = psycopg2.connect(database='wos_core_3',user="huanyong",password = "pendulum_wist_estival")
        self._cursor = self._db.cursor()


    def query_database(self,sql):
        self._cursor.close()
        self._cursor = self._db.cursor()
        self._cursor.execute(sql)
        logging.debug("query database with sql {:}".format(sql))
        return self._cursor

    def insert_database(self,sql,values):
        self._cursor.close()
        self._cursor = self._db.cursor()
        self._cursor.executemany(sql,values)
        logging.debug("insert data to database with sql {:}".format(sql))
        self._db.commit()


    def batch_insert(self,sql,row,step,is_auto=True,end=False):
        if end:
            if len(self._insert_values)!=0:
                logging.info("insert {:}th data into database,final insert.".format(self._insert_index))
                self.insert_database(sql,self._insert_values)

        else:
            self._insert_index+=1

            if is_auto:
                row[0] = self._insert_index

            self._insert_values.append(tuple(row))
            if self._insert_index%step==0:
                logging.info("insert {:}th data into database".format(self._insert_index))
                self.insert_database(sql,self._insert_values)
                self._insert_values=[]

    def get_insert_count(self):
        return self._insert_index

    def execute_del_update(self,sql):
        self._cursor.execute(sql)
        self._db.commit()
        logging.debug("execute delete or update sql {:}.".format(sql))

    def execute_sql(self,sql):
        self._cursor.execute(sql)
        self._db.commit()
        logging.debug("execute sql {:}.".format(sql))

    def close_db(self):
        self._db.close()


