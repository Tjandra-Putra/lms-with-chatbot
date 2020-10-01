from main import Messages

data = Messages.query.all()

for rows in data:
    print(rows.tag)