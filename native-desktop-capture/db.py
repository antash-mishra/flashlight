import sqlite3

class DB:
    def __init__(self):
        print("Initializing database...")
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for storing tab data"""
        self.conn = sqlite3.connect('tabs.db', check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tabs (
                id INTEGER PRIMARY KEY,
                tab_id TEXT,
                window_id TEXT,
                title TEXT,
                url TEXT,
                content TEXT,
                embedding BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
