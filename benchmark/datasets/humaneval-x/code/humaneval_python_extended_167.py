import psycopg2


def connect_postgres(host, port, username, password, database) -> psycopg2.extensions.connection:
    """
    Connect to a PostgreSQL database with the given parameters: host, port, username, password, and database.
    >>> conn = connect_postgres('localhost', 5432, 'postgres', 'password', 'postgres')
    """
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database
    )
    return conn


def check(connect_postgres):
        # Test connection parameters
    conn = connect_postgres(
            'localhost', 5432, 'postgres', 'password', 'postgres')
    assert conn is not None

    # Test connection by executing a simple query
    cur = conn.cursor()
    cur.execute('SELECT 1')
    assert cur.fetchone() == (1,)
    cur.close()
    conn.close()

check(connect_postgres)