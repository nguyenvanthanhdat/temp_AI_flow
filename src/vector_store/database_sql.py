import sqlite3
from typing import Any, Dict, List
import time

class DatabaseSQL:
    def __init__(self, databse_name: str, table_name: str) -> None:
        self.database_name = databse_name
        self.table_name = table_name
        self.connection = None

    def connect(self) -> None:
        """
            Connect to the MySQL database.
        """
        while self.connection is None:
            try:
                self.connection = sqlite3.connect(database=self.database_name)
            except sqlite3.Error as e:
                print(f"Error while connecting to SQLite: {e}")  
                print("Retrying in 5 seconds...")
            time.sleep(5)  

    def close(self) -> None:
        """
            Close the connection to the MySQL database.
        """
        if self.connection is not None:
            self.connection.close()
            print("Connection closed successfully")

    def get_cursor(self):
        return self.connection.cursor()

class ImageDatabaseSQL(DatabaseSQL):
    def __init__(self, database_name: str, table_name: str) -> None:
        super().__init__(database_name, table_name)
    
    def create_table(self) -> None:
        """
            Create a table in the MySQL database.
        """
        if self.connection is None:
            self.connection = self.connect()
        cursor = self.connection.cursor()
        cursor.execute(f"""
                       CREATE TABLE IF NOT EXISTS {self.table_name} (
                       video_id VARCHAR(255), 
                       keyframe_id VARCHAR(255), 
                       image BLOB,
                       PRIMARY KEY (video_id, keyframe_id)
                       )""")
        self.connection.commit()
        cursor.close()
        print(f"Table {self.table_name} created successfully")

    def insert(self, video_id: str, keyframe_id: str, image_path: str) -> None:
        """
            Insert content into the MySQL database with composite primary key (video_id, keyframe_id).
            The content is raw binary data.
        """
        if self.connection is None:
            self.connect()

        with open(image_path, 'rb') as file:
            binary_data = file.read()

        cursor = self.connection.cursor()
        cursor.execute(f"""
                       REPLACE INTO {self.table_name} (video_id, keyframe_id, image) 
                       VALUES (?, ?, ?)
                       """, (video_id, keyframe_id, binary_data))
        self.connection.commit()
        cursor.close()

class TextDatabaseSQL(DatabaseSQL):
    def __init__(self, database_name: str, table_name: str) -> None:
        super().__init__(database_name, table_name)

    def create_table(self) -> None:
        """
            Create a table in the MySQL database.
        """
        if self.connection is None:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(f"""
                       CREATE TABLE IF NOT EXISTS {self.table_name} (
                       video_id VARCHAR(255), 
                       keyframe_id VARCHAR(255), 
                       object TEXT,
                       character TEXT,
                       event TEXT,
                       scene TEXT,
                       summary TEXT,
                       PRIMARY KEY (video_id, keyframe_id)
                       )""")
        self.connection.commit()
        cursor.close()
        print(f"Table {self.table_name} created successfully")

    def insert(self, video_id: str, keyframe_id: str, metadata: List[str]) -> None:
        """
            Insert content into the MySQL database with composite primary key (video_id, keyframe_id).
            The content is list string of metadata
                - object
                - character
                - event TEXT,
                - scene TEXT,
                - summary TEXT,
        """
        if self.connection is None:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute(f"""
                       REPLACE INTO {self.table_name} (video_id, keyframe_id, object, character, event, scene, summary) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       """, (video_id, keyframe_id, metadata[0], metadata[1], metadata[2], metadata[3], metadata[4]))
        self.connection.commit()
        cursor.close()