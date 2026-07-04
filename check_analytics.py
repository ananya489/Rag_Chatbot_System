import sqlite3
import config


conn = sqlite3.connect(
    str(config.DB_PATH)
)

conn.row_factory = sqlite3.Row

cursor = conn.cursor()


print("Database:", config.DB_PATH)


cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
)

print("Tables:")
print(cursor.fetchall())


print("\nAnalytics:")

cursor.execute(
    "SELECT * FROM analytics"
)

rows = cursor.fetchall()


for row in rows:
    print(dict(row))


conn.close()