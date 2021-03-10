import psycopg2
import psycopg2.extras


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

    def createTables(self):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        # Doping EMPLOYEE table if already exists.
        cursor.execute("DROP TABLE IF EXISTS NON_LOCAL_GAMES_EVALUATED")

        # Creating table as per requirement
        sql = '''CREATE TABLE NON_LOCAL_GAMES_EVALUATED(
           id SERIAL PRIMARY KEY,
           QUESTIONS INT NOT NULL CHECK ( QUESTIONS >= 2 ),
           PLAYERS INT NOT NULL CHECK ( PLAYERS >= 2 ),
           CATEGORY FLOAT[] NOT NULL CHECK ( 0 <= ALL(CATEGORY) AND 1 >= ALL(CATEGORY) ),
           DIFFICULTY INT NOT NULL CHECK ( DIFFICULTY >= 0),
           MIN_CLASSIC_VALUE FLOAT NOT NULL CHECK ( MIN_CLASSIC_VALUE >= 0 AND MIN_CLASSIC_VALUE <= 1),
           MIN_QUANTUM_VALUE FLOAT NOT NULL CHECK ( MIN_QUANTUM_VALUE >= 0 AND MIN_QUANTUM_VALUE <= 1),
           MAX_CLASSIC_VALUE FLOAT NOT NULL CHECK ( MAX_CLASSIC_VALUE >= 0 AND MAX_CLASSIC_VALUE <= 1),
           MAX_QUANTUM_VALUE FLOAT NOT NULL CHECK ( MAX_QUANTUM_VALUE >= 0 AND MAX_QUANTUM_VALUE <= 1),
           MIN_DIFFERENCE FLOAT NOT NULL CHECK (MIN_DIFFERENCE >= 0 AND MIN_DIFFERENCE <= 1),
           MAX_DIFFERENCE FLOAT NOT NULL CHECK (MAX_DIFFERENCE >= 0 AND MAX_DIFFERENCE <= 1),
           MIN_STRATEGY TEXT[] NOT NULL,
           MAX_STRATEGY TEXT[] NOT NULL,
           MIN_STATE FLOAT[] NOT NULL,
           MAX_STATE FLOAT[] NOT NULL,
           GAME FLOAT[] NOT NULL,
           unique (PLAYERS, QUESTIONS, GAME)
        )'''

        cursor.execute(sql)
        print("Table created successfully........")

        # Doping EMPLOYEE table if already exists.
        cursor.execute("DROP TABLE IF EXISTS NON_LOCAL_GAMES_GENERATED")

        # Creating table as per requirement
        sql = '''CREATE TABLE NON_LOCAL_GAMES_GENERATED(
                   id SERIAL PRIMARY KEY,
                   QUESTIONS INT NOT NULL CHECK ( QUESTIONS >= 2 ),
                   PLAYERS INT NOT NULL CHECK ( PLAYERS >= 2 ),
                   CATEGORY FLOAT[] NOT NULL CHECK ( 0 <= ALL(CATEGORY) AND 1 >= ALL(CATEGORY) ),
                   DIFFICULTY INT NOT NULL CHECK ( DIFFICULTY >= 0),
                   GAME FLOAT[] NOT NULL,
                   unique (PLAYERS, QUESTIONS, GAME)
        )'''

        cursor.execute(sql)
        print("Table created successfully........")

        # Closing the connection
        conn.close()

    def query(self, category="all", difficulty="all", difference="all", num_players=2, n_questions=2):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        begin = '''SELECT * FROM NON_LOCAL_GAMES_EVALUATED '''
        if difference in ["max", "min"]:
            begin = '''SELECT DISTINCT ON (QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY) QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MIN_CLASSIC_VALUE, MIN_QUANTUM_VALUE, MAX_CLASSIC_VALUE, MAX_QUANTUM_VALUE, MIN_DIFFERENCE, MAX_DIFFERENCE, MIN_STATE, MAX_STATE, MIN_STRATEGY, MAX_STRATEGY GAME FROM NON_LOCAL_GAMES_EVALUATED '''

        if difficulty == "all" and category == "all":  sql = begin
        elif difficulty == "all": sql = begin + '''WHERE ''' + '''CATEGORY = ''' + str(category) + ''' AND PLAYERS = ''' + str(
            num_players) + ''' AND QUESTIONS = ''' + str(n_questions)
        else: sql = begin + '''WHERE ''' + '''CATEGORY = ''' + str(category) + ''' AND DIFFICULTY = ''' + str(
            difficulty) + ''' AND PLAYERS = ''' + str(num_players) + ''' AND QUESTIONS = ''' + str(n_questions)

        if difference == "max":
            sql += '''ORDER BY QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MAX_DIFFERENCE DESC''';
        if difference == "min":
            sql += '''ORDER BY QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MAX_DIFFERENCE ASC''';

        # Retrieving data
        cursor.execute(sql)

        result = cursor.fetchall();

        print("Records queried")

        # Closing the connection
        conn.close()

        return result

    def query_categories_games(self, num_players=2, n_questions=2):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        sql = '''SELECT category,difficulty, game FROM NON_LOCAL_GAMES_GENERATED 
        WHERE ''' + ''' PLAYERS = ''' + str(num_players) + ''' AND QUESTIONS = ''' + str(n_questions)

        # Retrieving data
        cursor.execute(sql)

        result = cursor.fetchall();

        print("Records queried")

        # Closing the connection
        conn.close()

        return result

    def insert_categories_games(self, n_questions, num_players, generated_games):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        for category, difficulties in generated_games.items():
            for difficulty, games in difficulties.items():
                for game in games:
                    sql = "INSERT INTO NON_LOCAL_GAMES_GENERATED ( PLAYERS, QUESTIONS, DIFFICULTY, CATEGORY, GAME) VALUES (" + str(
                        num_players) + ", " + str(n_questions) + ", " + str(difficulty) + ", ARRAY[" + str(list(category)) + "], ARRAY[" + str(
                        game) + "]);" \
                                ""
                    cursor.execute(sql)

                    # Commit your changes in the database
                    conn.commit()

        print("Records inserted")

    def insert(self, category, difficulty, classic_min, quantum_min, classic_max, quantum_max, difference_min, difference_max, min_state, max_state,
               min_strategy, max_strategy, game, questions=2,
               players=2):
        # establishing the connection
        conn = psycopg2.connect(
            database="postgres", user='postgres', password='password', host='127.0.0.1', port='5432'
        )
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()

        sql = '''INSERT INTO NON_LOCAL_GAMES_EVALUATED(QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MIN_CLASSIC_VALUE, MIN_QUANTUM_VALUE, MAX_CLASSIC_VALUE, MAX_QUANTUM_VALUE, MIN_DIFFERENCE, MAX_DIFFERENCE, MIN_STATE, MAX_STATE, MIN_STRATEGY, MAX_STRATEGY, GAME) VALUES ( ''' + str(
            questions) + "," + str(
            players) + ", ARRAY[" + str(category) + "]," + str(difficulty) + "," + str(classic_min) + "," + str(quantum_min) + "," + str(
            classic_max) + "," + str(quantum_max) + "," + str(
            difference_min) + "," + str(
            difference_max) + ",ARRAY[" + str(
            min_state) + "], ARRAY[" + str(
            max_state) + "], ARRAY[" + str(
            min_strategy) + "], ARRAY[" + str(
            max_strategy) + "], ARRAY[" + str(game) + '''] )
            ON CONFLICT(PLAYERS, QUESTIONS, GAME) DO 
            UPDATE SET MAX_QUANTUM_VALUE = EXCLUDED.MAX_QUANTUM_VALUE, MAX_DIFFERENCE = EXCLUDED.MAX_DIFFERENCE, MAX_STATE = EXCLUDED.MAX_STATE, MAX_STRATEGY = EXCLUDED.MAX_STRATEGY
            WHERE EXCLUDED.MAX_QUANTUM_VALUE > NON_LOCAL_GAMES_EVALUATED.MAX_QUANTUM_VALUE;
            '''

        cursor.execute(sql)

        # Commit your changes in the database
        conn.commit()

        sql = '''INSERT INTO NON_LOCAL_GAMES_EVALUATED(QUESTIONS, PLAYERS, CATEGORY, DIFFICULTY, MIN_CLASSIC_VALUE, MIN_QUANTUM_VALUE, MAX_CLASSIC_VALUE, MAX_QUANTUM_VALUE, MIN_DIFFERENCE, MAX_DIFFERENCE, MIN_STATE, MAX_STATE, MIN_STRATEGY, MAX_STRATEGY, GAME) VALUES ( ''' + str(
            questions) + "," + str(
            players) + ", ARRAY[" + str(category) + "]," + str(difficulty) + "," + str(classic_min) + "," + str(quantum_min) + "," + str(
            classic_max) + "," + str(quantum_max) + "," + str(
            difference_min) + "," + str(
            difference_max) + ",ARRAY[" + str(
            min_state) + "], ARRAY[" + str(
            max_state) + "], ARRAY[" + str(
            min_strategy) + "], ARRAY[" + str(
            max_strategy) + "], ARRAY[" + str(game) + '''] )
            ON CONFLICT(PLAYERS, QUESTIONS, GAME) DO 
            UPDATE SET MIN_QUANTUM_VALUE = EXCLUDED.MIN_QUANTUM_VALUE, MIN_DIFFERENCE = EXCLUDED.MIN_DIFFERENCE, MIN_STATE = EXCLUDED.MIN_STATE, MIN_STRATEGY = EXCLUDED.MIN_STRATEGY
            WHERE EXCLUDED.MIN_QUANTUM_VALUE < NON_LOCAL_GAMES_EVALUATED.MIN_QUANTUM_VALUE;
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

    db.createTables()
    # db.insert(category=1, difficulty=2, difference_max=1, difference_min=1, questions=2, players=2, classic_min=1, quantum_min=1, classic_max=1,
    #           quantum_max=1, game=[[9]])
    # db.insert(category=1, difficulty=2, difference=1, questions=2, players=2, classic=1, quantum=0.75, game=[[8]])
    # print(db.query(max_difference=True))
    print(db.query())
