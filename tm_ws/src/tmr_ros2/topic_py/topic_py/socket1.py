#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import mysql.connector
import time

import socket
HOST = '192.168.10.10'
PORT = 5000
i=0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind((HOST, PORT))
s.listen(5)
scan_num=0
encode=["T101","T102","T201","T202","T301"]
name=["ram1","ram2","cpu1","cpu2","board"]
print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')
conn, addr = s.accept()
print('connecting by ' + str(addr))
ram=[]
cpu=[]
# conn, addr = s.accept()
# print('connecting by ' + str(addr))
while True:
    indata = conn.recv(1024)
    code=indata.decode()
    if code!='':
        print('recv: ' + indata.decode())

        # outdata = 'echo ' + indata.decode()
        # conn.send(outdata.encode())
        # conn.close()
        # s.close()   
    
        conexion = mysql.connector.connect(
            host='localhost',
            user='root',
            passwd='',
            database='tm_test'
        )

        # print(code[0:3])
        for i in range(3):
            if code[0:4]== "T301" and scan_num==3:
                print("AAAA")
                cur01 = conexion.cursor()
                t=time.localtime()
                time1=str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)+' '+str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)
                # time1[str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)]=str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)
                insertar = "INSERT INTO scanning VALUES (%s, %s,%s,%s,%s)"
                a=(code,"board",time1,"out",code[4:7])
                cur01.execute(insertar,a)
                conexion.commit()

                insertar="select num from holdings where name=%s"
                name_tuple="board",
                # print(type(name_tuple))
                cur01.execute(insertar,name_tuple)
                data=cur01.fetchone()
                # print(data)
                data=int(data[0])
                data-=1
                update_data=(data,"board")
                insertar = "UPDATE holdings SET num=%s where name=%s"
                cur01.execute(insertar,update_data)
                conexion.commit()
                scan_num+=1
                
                t=time.localtime()
                time1=str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)+' '+str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)
                insertar = "INSERT INTO done VALUES (%s, %s,%s,%s,%s,%s,%s)"
                finish_item=(code,"board",ram[0],ram[1],cpu[0],time1,code[4:7])
                cur01.execute(insertar,finish_item)
                conexion.commit()
                break





            if code[0:4] == encode[i]:
                print("QQQQ")
                if code[1:4]=="101":
                    ram.append("ram1")

                elif code[1:4]=="102":
                    ram.append("ram2")

                elif code[1:4]=="201":
                    cpu.append("cpu1")

                elif code[1:4]=="202":
                    cpu.append("cpu2")
                print(name[i])
                cur01 = conexion.cursor()
                t=time.localtime()
                time1=str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)+' '+str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)
                # time1[str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)]=str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)
                insertar = "INSERT INTO scanning VALUES (%s, %s,%s,%s,%s)"
                a=(code,name[i],time1,"out",code[4:7])
                cur01.execute(insertar,a)
                conexion.commit()

                insertar="select num from holdings where name=%s"
                name_tuple=name[i],
                # print(type(name_tuple))
                cur01.execute(insertar,name_tuple)
                data=cur01.fetchone()
                # print(data)
                data=int(data[0])
                data-=1
                update_data=(data,name[i])
                insertar = "UPDATE holdings SET num=%s where name=%s"
                cur01.execute(insertar,update_data)
                conexion.commit()
                scan_num+=1

        

        # del
        # mycursor = conexion.cursor()

        # sql = "DELETE FROM tm_database WHERE num = '%s'"
        # a=[]
        # a.append(i)
        # mycursor.execute(sql,a)
        
        # conexion.commit()
        
        # print(mycursor.rowcount, " data were delet")



        conexion.close()
        i+=1
    # else:
    #   s.close() 