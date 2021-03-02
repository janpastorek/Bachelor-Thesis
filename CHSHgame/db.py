import psycopg2


class CHSHdb:

    def createDB(self):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        # Preparing query to create a database
        sql = '''CREATE database mydb''';

        # Creating a database
        cursor.execute(sql)
        print("Database created successfully........")

        # Closing the connection
        conn.close()

    def createTable(self):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        # Doping EMPLOYEE table if already exists.
        cursor.execute("DROP TABLE IF EXISTS CHSH")

        # Creating table as per requirement
        sql = '''CREATE TABLE CHSH(
           id SERIAL PRIMARY KEY,
           QUESTIONS INT NOT NULL CHECK ( QUESTIONS >= 2 ),
           PLAYERS INT NOT NULL CHECK ( PLAYERS >= 2 ),
           CATEGORY FLOAT NOT NULL CHECK ( CATEGORY >= 0 AND CATEGORY <= 1),
           DIFFICULTY INT NOT NULL CHECK ( DIFFICULTY >= 0),
           CLASSIC_VALUE FLOAT NOT NULL CHECK ( CLASSIC_VALUE >= 0 AND CLASSIC_VALUE <= 1),
           QUANTUM_VALUE FLOAT NOT NULL CHECK ( QUANTUM_VALUE >= 0 AND QUANTUM_VALUE <= 1),
           DIFFERENCE FLOAT NOT NULL,
           GAME FLOAT[] NOT NULL UNIQUE
        )'''

        cursor.execute(sql)
        print("Table created successfully........")

        # Closing the connection
        conn.close()

    def query(self, category="all", difficulty="all", max_difference=False):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        begin = '''SELECT * FROM CHSH '''
        if max_difference:
            begin = '''SELECT DISTINCT ON (QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY) QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY,CLASSIC_VALUE, QUANTUM_VALUE, DIFFERENCE, GAME FROM CHSH '''

        if difficulty == "all" and category == "all":  sql = begin
        elif difficulty == "all": sql = begin + '''WHERE ''' + '''CATEGORY = ''' + str(category)
        else: sql = begin + '''WHERE ''' + '''CATEGORY = ''' + str(category) + ''' AND DIFFICULTY = ''' + str(difficulty)

        sql += '''ORDER BY QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, DIFFERENCE DESC''';

        # Retrieving data
        cursor.execute(sql)

        result = cursor.fetchall();

        print("Records queried")

        # Closing the connection
        conn.close()

        return result

    def insert(self, category, difficulty, classic, quantum, difference, game, questions=2, players=2):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        sql = '''INSERT INTO CHSH(QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, CLASSIC_VALUE, QUANTUM_VALUE, DIFFERENCE, GAME) VALUES ( ''' + str(
            questions) + "," + str(
            players) + "," + str(category) + "," + str(difficulty) + "," + str(classic) + "," + str(quantum) + "," + str(
            difference) + ", ARRAY[" + str(game) + '''] )
            ON CONFLICT(GAME) DO 
            UPDATE SET QUANTUM_VALUE = EXCLUDED.QUANTUM_VALUE
            WHERE EXCLUDED.QUANTUM_VALUE > CHSH.QUANTUM_VALUE;
            '''

        cursor.execute(sql)

        # Commit your changes in the database
        conn.commit()
        print("Record inserted")

        # Closing the connection
        conn.close()


if __name__ == '__main__':
    db = CHSHdb()

    # db.createDB()

    # db.createTable()
    # db.insert(category=1, difficulty=2, difference=1, questions=2, players=2, classic=1, quantum=1, game=[[9]])
    # db.insert(category=1, difficulty=2, difference=1, questions=2, players=2, classic=1, quantum=0.75, game=[[8]])
    print(db.query(max_difference=True))
    print(db.query())
