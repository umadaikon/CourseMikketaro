import mysql.connector

conn = mysql.connector.connect(
	user='root',
	password='',
	host='localhost',
	database='bouldering_db'
)
cur = conn.cursor()

#move = ["deadpoint","dyagonal"]
move = "cross"

#list1 = [1,2,3,4,5]
#count = 1
'''
if move == "deadpoint":
	for counter in list1:
		count += 1
'''

id = 1
# distance = 5
# angle = 45

value = "values(" + str(id) + ", " + str(p_beforex) + ", " + str(p_beforey) + ", " + str(b_ax) + ", " + str(b_ay) + ", " + str(mainhand) + ");"
i_command = "insert into " + move + "(id, p_distance, p_angle, m_distance, m_angle, mainhand) " + value
s_command = "select * from " + move + ";"

#print (value)
print (i_command)
print (s_command)
cur.execute(i_command)

#for row in cur.fetchall():
#	print(row[0],row[1],row[2],row[3],row[4])

#print("deadpoint")

cur.close()
conn.commit()
conn.close()