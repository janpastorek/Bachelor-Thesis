import psycopg2


class CHSHdb:

    def __init__(self):
        pass

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
           MIN_CLASSIC_VALUE FLOAT NOT NULL CHECK ( MIN_CLASSIC_VALUE >= 0 AND MIN_CLASSIC_VALUE <= 1),
           MIN_QUANTUM_VALUE FLOAT NOT NULL CHECK ( MIN_QUANTUM_VALUE >= 0 AND MIN_QUANTUM_VALUE <= 1),
           MAX_CLASSIC_VALUE FLOAT NOT NULL CHECK ( MAX_CLASSIC_VALUE >= 0 AND MAX_CLASSIC_VALUE <= 1),
           MAX_QUANTUM_VALUE FLOAT NOT NULL CHECK ( MAX_QUANTUM_VALUE >= 0 AND MAX_QUANTUM_VALUE <= 1),
           MIN_DIFFERENCE FLOAT NOT NULL,
           MAX_DIFFERENCE FLOAT NOT NULL,
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
            begin = '''SELECT DISTINCT ON (QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY) QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MIN_CLASSIC_VALUE, MIN_QUANTUM_VALUE, MAX_CLASSIC_VALUE, MAX_QUANTUM_VALUE, MIN_DIFFERENCE, MAX_DIFFERENCE, GAME FROM CHSH '''

        if difficulty == "all" and category == "all":  sql = begin
        elif difficulty == "all": sql = begin + '''WHERE ''' + '''CATEGORY = ''' + str(category)
        else: sql = begin + '''WHERE ''' + '''CATEGORY = ''' + str(category) + ''' AND DIFFICULTY = ''' + str(difficulty)

        sql += '''ORDER BY QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MAX_DIFFERENCE DESC''';

        # Retrieving data
        cursor.execute(sql)

        result = cursor.fetchall();

        print("Records queried")

        # Closing the connection
        conn.close()

        return result

    def insert(self, category, difficulty, classic_min, quantum_min, classic_max, quantum_max, difference_min, difference_max, game, questions=2,
               players=2):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        sql = '''INSERT INTO CHSH(QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MIN_CLASSIC_VALUE, MIN_QUANTUM_VALUE, MAX_CLASSIC_VALUE, MAX_QUANTUM_VALUE, MIN_DIFFERENCE, MAX_DIFFERENCE, GAME) VALUES ( ''' + str(
            questions) + "," + str(
            players) + "," + str(category) + "," + str(difficulty) + "," + str(classic_min) + "," + str(quantum_min) + "," + str(
            classic_max) + "," + str(quantum_max) + "," + str(
            difference_min) + "," + str(
            difference_max) + ", ARRAY[" + str(game) + '''] )
            ON CONFLICT(GAME) DO 
            UPDATE SET MAX_QUANTUM_VALUE = EXCLUDED.MAX_QUANTUM_VALUE, MAX_DIFFERENCE = EXCLUDED.MAX_DIFFERENCE
            WHERE EXCLUDED.MAX_QUANTUM_VALUE > CHSH.MAX_QUANTUM_VALUE;
            '''

        cursor.execute(sql)

        # Commit your changes in the database
        conn.commit()

        sql = '''INSERT INTO CHSH(QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MIN_CLASSIC_VALUE, MIN_QUANTUM_VALUE, MAX_CLASSIC_VALUE, MAX_QUANTUM_VALUE, MIN_DIFFERENCE, MAX_DIFFERENCE, GAME) VALUES ( ''' + str(
            questions) + "," + str(
            players) + "," + str(category) + "," + str(difficulty) + "," + str(classic_min) + "," + str(quantum_min) + "," + str(
            classic_max) + "," + str(quantum_max) + "," + str(
            difference_min) + "," + str(
            difference_max) + ", ARRAY[" + str(game) + '''] )
                    ON CONFLICT(GAME) DO 
                    UPDATE SET MIN_QUANTUM_VALUE = EXCLUDED.MIN_QUANTUM_VALUE, MIN_DIFFERENCE = EXCLUDED.MIN_DIFFERENCE
                    WHERE EXCLUDED.MIN_QUANTUM_VALUE < CHSH.MIN_QUANTUM_VALUE;
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
