import sqlite3

conn = sqlite3.connect("mathsite.db")
c = conn.cursor()

print("Users in database:")
for row in c.execute("SELECT id, username FROM users"):
    print(row)

conn.close()