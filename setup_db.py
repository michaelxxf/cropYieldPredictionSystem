import mysql.connector
from mysql.connector import Error

def setup_database():
    # First connection to create database
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""  # Update this if you have set a password
        )
        
        if conn.is_connected():
            cursor = conn.cursor()
            
            # Create database
            cursor.execute("CREATE DATABASE IF NOT EXISTS crop_prediction")
            print("Database created successfully")
            
            # Switch to the database
            cursor.execute("USE crop_prediction")
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    email VARCHAR(100) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Users table created successfully")
            
            # Drop existing predictions table if it exists (to update schema)
            cursor.execute("DROP TABLE IF EXISTS predictions")
            
            # Create predictions table with created_at column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    nitrogen FLOAT NOT NULL,
                    phosphorus FLOAT NOT NULL,
                    potassium FLOAT NOT NULL,
                    temperature FLOAT NOT NULL,
                    humidity FLOAT NOT NULL,
                    ph FLOAT NOT NULL,
                    rainfall FLOAT NOT NULL,
                    yield_prediction FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            print("Predictions table created successfully")
            
            # Create crop_facts table
            cursor.execute("DROP TABLE IF EXISTS crop_facts")
            cursor.execute("""
                CREATE TABLE crop_facts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(100) NOT NULL,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Crop facts table created successfully")
            
            # Insert sample crop facts
            sample_facts = [
                ('Soil pH Matters', 'Different crops thrive in different pH levels. Most crops prefer slightly acidic soil with pH between 6.0 and 6.8.'),
                ('NPK Balance', 'Nitrogen (N) promotes leaf growth, Phosphorus (P) helps root and flower development, and Potassium (K) improves overall plant health.'),
                ('Water Management', 'While water is essential for crop growth, overwatering can be as harmful as underwatering. Good drainage is crucial.'),
                ('Temperature Impact', 'Each crop has an optimal temperature range. Most crops grow best between 20°C and 30°C.'),
                ('Humidity Effects', 'High humidity can increase the risk of fungal diseases, while low humidity can increase water loss through transpiration.')
            ]
            
            cursor.executemany(
                "INSERT INTO crop_facts (title, description) VALUES (%s, %s)",
                sample_facts
            )
            print("Sample crop facts inserted successfully")
            
            conn.commit()
            print("All database setup completed successfully!")
            
    except Error as e:
        print(f"Error: {e}")
        
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection closed.")

if __name__ == "__main__":
    setup_database() 