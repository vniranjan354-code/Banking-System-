import mysql.connector

def setup_database():
    try:
        # Connect without database to create it
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        cursor = conn.cursor()

        # Drop and Recreate Database
        print("Dropping database facespoofbanking_2026 if exists...")
        cursor.execute("DROP DATABASE IF EXISTS facespoofbanking_2026")
        print("Creating database facespoofbanking_2026...")
        cursor.execute("CREATE DATABASE facespoofbanking_2026")
        cursor.execute("USE facespoofbanking_2026")

        # Create Banks Table
        print("Creating table: banks...")
        cursor.execute("""
            CREATE TABLE banks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bank_name VARCHAR(100) NOT NULL,
                branch_name VARCHAR(100) NOT NULL,
                ifsc VARCHAR(20) NOT NULL,
                address TEXT NOT NULL
            )
        """)

        # Create Accounts Table
        print("Creating table: accounts...")
        cursor.execute("""
            CREATE TABLE accounts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(20) NOT NULL UNIQUE,
                bank_name VARCHAR(100) NOT NULL,
                branch_name VARCHAR(100) NOT NULL,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(50) NOT NULL,
                acc_num VARCHAR(20) NOT NULL UNIQUE,
                balance DECIMAL(15, 2) DEFAULT 0.00,
                full_name VARCHAR(100) NOT NULL,
                age INT NOT NULL,
                email VARCHAR(100) NOT NULL,
                gender VARCHAR(20) NOT NULL,
                phone VARCHAR(20) NOT NULL
            )
        """)

        # Create History Table
        print("Creating table: history...")
        cursor.execute("""
            CREATE TABLE history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tran_id VARCHAR(50) NOT NULL UNIQUE,
                from_acc VARCHAR(20) NOT NULL,
                to_acc VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                amount DECIMAL(15, 2) NOT NULL,
                type VARCHAR(20) NOT NULL
            )
        """)

        # Seed initial data for Banks
        print("Seeding initial data for banks...")
        banks_data = [
            ('SafeBank', 'Main Branch', 'SAFE0001234', '123 Finance St, City'),
            ('TrustBank', 'Downtown', 'TRST0005678', '456 Commerce Ave, City'),
            ('UniBank', 'North Branch', 'UNIB0009012', '789 Central Rd, City')
        ]
        cursor.executemany("INSERT INTO banks (bank_name, branch_name, ifsc, address) VALUES (%s, %s, %s, %s)", banks_data)

        conn.commit()
        print("Database and tables created successfully!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    setup_database()
