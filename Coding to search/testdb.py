import psycopg2

# Defining database connection parameters
db_params = {
    "host": "riteh-pg-sec.int.ototrak.com",
    "database": "ototrak",
    "user": "riteh",
    "password": "choopiphohp1Aivah7ud",
    "port": "5432"
}

try:
    print("Connection established")
    # Establishing a connection to the database
    connection = psycopg2.connect(**db_params)

    # Performing database operations here...
    print("Connection established2")

    # Defining the SELECT query
    select_query = "SELECT * FROM tours WHERE device_id = 5787;"

    print("Connection established3")
    # Creating a cursor object to interact with the database 
    cursor = connection.cursor()
    print("Connection established4")

    # Executing the SELECT query
    cursor.execute(select_query)

    # Fetching and printing the results
    result = cursor.fetchall()
    for row in result:
        print(row)
except (Exception, psycopg2.Error) as error:
    print("Connection no")
    print(f"Error connecting to the database: {error}")

finally:
    print("Connection end")
    if connection:
        cursor.close()
        connection.close()
        print("Database connection closed.")
