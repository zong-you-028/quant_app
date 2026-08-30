"""Journal database access: Neon PostgreSQL in production, SQLite locally."""
import os


class _PostgresConnection:
    dialect = "postgres"

    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql, params=()):
        return self._connection.execute(sql.replace("?", "%s"), params)

    def commit(self):
        self._connection.commit()

    def close(self):
        self._connection.close()


def get_journal_conn():
    url = os.environ.get("JOURNAL_DATABASE_URL", "").strip()
    if url:
        import psycopg
        return _PostgresConnection(psycopg.connect(url))

    from core.data_pipeline import get_conn
    return get_conn()


def dialect(conn) -> str:
    return getattr(conn, "dialect", "sqlite")


def insert_id(conn, sql: str, params) -> int:
    if dialect(conn) == "postgres":
        row = conn.execute(sql.rstrip().rstrip(";") + " RETURNING id", params).fetchone()
        return int(row[0])
    return int(conn.execute(sql, params).lastrowid)
