from database_sql import ImageDatabaseSQL, TextDatabaseSQL
import os
from tqdm import tqdm

def main():

    os.makedirs("database_sql", exist_ok=True)
    image_database = ImageDatabaseSQL("example/sql_db/image.sqlite", "image")
    image_database.connect()

    text_database = TextDatabaseSQL("example/sql_db/text.sqlite", "text")
    text_database.connect()
    
    # Open if not have sql file
    # import json
    # with open('example/json/texts.json', 'r') as file:
    #     metadata_samples = json.load(file)
    # with open('example/json/images.json', 'r') as file:
    #     image_samples = json.load(file)

    # if len(image_samples) != len(metadata_samples):
    #     raise ValueError("The number of image samples and metadata samples must be the same")
    # image_database.create_table()
    # text_database.create_table()
    # for each_image, each_text in tqdm(zip(image_samples, metadata_samples), total=len(image_samples), desc="Inserting data"):
    #     image_database.insert(each_image["video_id"], each_image["keyframe_id"], each_image["image"])
    #     text_database.insert(each_text["video_id"], each_text["keyframe_id"], each_text["metadata"])

    print("Completed connected")

    print("Query on images table:")
    query_for_images = "SELECT keyframe_id FROM image WHERE video_id='1'"
    cursor_image = image_database.get_cursor()
    cursor_image.execute(query_for_images)
    result = cursor_image.fetchall()
    print(result)

    print("Query on texts table:")
    query_for_text = "SELECT keyframe_id FROM text WHERE video_id='1'"
    cursor_text = text_database.get_cursor()
    cursor_text.execute(query_for_text)
    result = cursor_text.fetchall()
    print(result)

if __name__ == "__main__":
    main()