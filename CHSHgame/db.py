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

        # Doping EMPLOYEE table if already exists.
        cursor.execute("DROP TABLE IF EXISTS CHSH")

        # TODO: dokoncit sql databazu

        # Creating table as per requirement
        sql = '''CREATE TABLE CHSH(
         CATEGORY FLOAT NOT NULL,
         DIFFICULTY INT NOT NULL,
         DIFFERENCE FLOAT NOT NULL,
         GAME CHAR(100) NOT NULL
      )'''

        cursor.execute(sql)
        print("Table created successfully........")

        # Closing the connection
        conn.close()

    def query(self, category, difficulty="all"):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        if difficulty == "all": sql = '''SELECT * FROM CHSH WHERE ''' + '''CATEGORY = ''' + category
        else: sql = '''SELECT * FROM CHSH WHERE ''' + '''CATEGORY = ''' + category + ''' AND DIFFICULTY = ''' + difficulty

        # Retrieving data
        cursor.execute(sql)

        result = cursor.fetchall();

        print("Record ")

        # Closing the connection
        conn.close()

        return result

    def insert(self, category, difficulty, difference, game):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        sql = '''INSERT INTO CHSH(CATEGORY, DIFFICULTY, DIFFERENCE, GAME,
         INCOME) VALUES ( ''' + category + "," + difficulty + "," + difference + "," + game + ''' )'''

        cursor.execute(sql)

        # Commit your changes in the database
        conn.commit()
        print("Record inserted")

        # Closing the connection
        conn.close()
