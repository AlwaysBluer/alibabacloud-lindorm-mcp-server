import mysql.connector
from mysql.connector import Error


class LindormWideTableClient:
    def __init__(self, table_host: str, username: str, password: str, database='default'):
        self.config = {
            'host': table_host,
            'port': 33060,
            'user': username,
            'password': password,
            'database': database
        }
        self.connection = None
        self.cursor = None
        self._connect()

    def _connect(self):
        self.connection = mysql.connector.connect(**self.config)
        self.cursor = self.connection.cursor()

    def __del__(self):
        self._close()

    def _close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> str:
        """Execute SQL commands."""
        try:
            self.cursor.execute(query)

            # Special handling for SHOW TABLES
            if query.strip().upper().startswith("SHOW TABLES"):
                tables = self.cursor.fetchall()
                result = ["Tables_in_" + self.config["database"]]  # Header
                result.extend([table[0] for table in tables])
                return "\n".join(result)

            # Regular SELECT queries
            elif query.strip().upper().startswith("SELECT") or query.strip().upper().startswith(
                    "SHOW") or query.strip().upper().startswith("DESCRIBE"):
                columns = [desc[0] for desc in self.cursor.description]
                rows = self.cursor.fetchall()
                result = [",".join(map(str, row)) for row in rows]
                return "\n".join([",".join(columns)] + result)

            # Non-SELECT queries
            else:
                self.connection.commit()
                return f"Query executed successfully. Rows affected: {self.cursor.rowcount}"

        except Error as e:
            error_msg = str(e)
            if "Detect inefficient query" in error_msg:
                return ("Your query was identified as inefficient. " +
                        "Please add /*+ _l_allow_filtering_ */ hint after the SELECT keyword.\n" +
                        "Example: SELECT /*+ _l_allow_filtering_ */ * FROM table\n" +
                        "Instead of: SELECT * FROM table")
            elif "JOIN is not allowed" in error_msg or "UNION is not allowed" in error_msg:
                return "JOIN UNION is not allowed. Please execute 'ALTER SYSTEM SET `lindorm.sql.join_union.disabled`=FALSE' to enable join."
            return f"Error executing query: {str(e)}"

    def reconnect(self):
        """Reconnect to the database."""
        self._close()
        self._connect()
