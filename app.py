import os
import random
import string
from datetime import date
from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from simple_auth import SimpleVideoAuthenticator
import tempfile

app = Flask(__name__)
app.secret_key = 'banking_secret_key_2026'
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

# Biometric Authenticator
auth = SimpleVideoAuthenticator()

# Database Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facespoofbanking_2026'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def save_upload(file) -> str:
    """Save uploaded video blob to a temp file and return the path."""
    suffix = ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    return tmp.name

# Helper functions
def generate_user_id():
    return f"User{random.randint(100, 999)}"

def generate_acc_num(bank_name):
    prefix = bank_name[:3].upper() if bank_name else "BNK"
    return f"{prefix}{random.randint(10000, 99999)}"

def generate_tran_id():
    return f"Tran{random.randint(10000, 99999)}"

# Email Configuration
EMAIL_USER = 'bookswisesage@gmail.com'
EMAIL_PASS = 'bivkblevyuattedr'

def send_transaction_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, recipient_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

def get_user_email(acc_num):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT email, full_name FROM accounts WHERE acc_num = %s", (acc_num,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    if request.method == 'POST':
        full_name = request.form['full_name']
        age = request.form['age']
        email = request.form['email']
        gender = request.form['gender']
        phone = request.form['phone']
        bank_name = request.form['bank_name']
        branch_name = request.form['branch_name']
        username = request.form['username']
        password = request.form['password']
        
        user_id = generate_user_id()
        acc_num = generate_acc_num(bank_name)
        
        # Biometric Enrollment
        if 'video' not in request.files:
            flash('Biometric video is required.', 'danger')
            return redirect(url_for('register'))
            
        video_path = save_upload(request.files['video'])
        
        try:
            # 1. Enroll Biometrics
            bio_result = auth.enroll(user_id, video_path)
            if not bio_result['success']:
                flash(f"Biometric enrollment failed: {bio_result['message']}", 'danger')
                return redirect(url_for('register'))
                
            # 2. Save to Database
            query = """INSERT INTO accounts (user_id, bank_name, branch_name, username, password, acc_num, balance, full_name, age, email, gender, phone) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(query, (user_id, bank_name, branch_name, username, password, acc_num, 0.00, full_name, age, email, gender, phone))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            flash(f'Error: {err}', 'danger')
        except Exception as e:
            flash(f'Biometric Error: {str(e)}', 'danger')
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            cursor.close()
            conn.close()
            
    # GET: Fetch banks for dropdown
    cursor.execute("SELECT * FROM banks")
    banks = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('register.html', banks=banks)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        acc_num = request.form['acc_num']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM accounts WHERE username = %s AND password = %s AND acc_num = %s"
        cursor.execute(query, (username, password, acc_num))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            # Biometric Authentication
            if 'video' not in request.files:
                flash('Biometric video is required for login.', 'danger')
                return render_template('login.html')
                
            video_path = save_upload(request.files['video'])
            
            try:
                bio_result = auth.authenticate(user['user_id'], video_path)
                
                if bio_result['authenticated']:
                    session['user_id'] = user['user_id']
                    session['username'] = user['username']
                    session['acc_num'] = user['acc_num']
                    flash('Login successful!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash(f"Biometric verification failed: {bio_result['message']}", 'danger')
            except Exception as e:
                flash(f"Biometric Error: {str(e)}", 'danger')
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM accounts WHERE user_id = %s", (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('dashboard.html', user=user)

@app.route('/transfer', methods=['GET', 'POST'])
def transfer():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    if request.method == 'POST':
        to_acc = request.form['to_acc']
        amount = float(request.form['amount'])
        
        # Check sender's balance
        cursor.execute("SELECT balance FROM accounts WHERE acc_num = %s", (session['acc_num'],))
        row = cursor.fetchone()
        if row and row['balance'] >= amount:
            try:
                # Deduct from sender
                cursor.execute("UPDATE accounts SET balance = balance - %s WHERE acc_num = %s", (amount, session['acc_num']))
                # Add to receiver
                cursor.execute("UPDATE accounts SET balance = balance + %s WHERE acc_num = %s", (amount, to_acc))
                
                # Record in history — Debit for sender, Credit for receiver
                tran_id = generate_tran_id()
                query = "INSERT INTO history (tran_id, from_acc, to_acc, date, amount, type) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(query, (tran_id, session['acc_num'], to_acc, date.today(), amount, 'Debit'))
                tran_id_credit = generate_tran_id()
                cursor.execute(query, (tran_id_credit, session['acc_num'], to_acc, date.today(), amount, 'Credit'))
                
                conn.commit()
                
                # Send Emails
                sender_info = get_user_email(session['acc_num'])
                receiver_info = get_user_email(to_acc)
                
                if sender_info:
                    send_transaction_email(
                        sender_info['email'],
                        "Transaction Alert: Debit",
                        f"Hello {sender_info['full_name']},\n\nYour account {session['acc_num']} has been debited by ₹{amount:,.2f} for a transfer to account {to_acc}.\n\nTransaction ID: {tran_id}\nDate: {date.today()}"
                    )
                
                if receiver_info:
                    send_transaction_email(
                        receiver_info['email'],
                        "Transaction Alert: Credit",
                        f"Hello {receiver_info['full_name']},\n\nYour account {to_acc} has been credited with ₹{amount:,.2f} from account {session['acc_num']}.\n\nTransaction ID: {tran_id_credit}\nDate: {date.today()}"
                    )

                flash('Transfer successful!', 'success')
                return redirect(url_for('dashboard'))
            except mysql.connector.Error as err:
                conn.rollback()
                flash(f'Transaction failed: {err}', 'danger')
        else:
            flash('Insufficient balance.', 'warning')
            
    # GET: List other accounts
    cursor.execute("SELECT acc_num, username FROM accounts WHERE acc_num != %s", (session['acc_num'],))
    accounts = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('transfer.html', accounts=accounts)

@app.route('/withdraw', methods=['GET', 'POST'])
def withdraw():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        amount = float(request.form['amount'])
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT balance FROM accounts WHERE acc_num = %s", (session['acc_num'],))
        row = cursor.fetchone()
        if row and row['balance'] >= amount:
            try:
                cursor.execute("UPDATE accounts SET balance = balance - %s WHERE acc_num = %s", (amount, session['acc_num']))
                tran_id = generate_tran_id()
                query = "INSERT INTO history (tran_id, from_acc, to_acc, date, amount, type) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(query, (tran_id, session['acc_num'], 'Self', date.today(), amount, 'Withdrawal'))
                conn.commit()
                
                # Send Email
                user_info = get_user_email(session['acc_num'])
                if user_info:
                    send_transaction_email(
                        user_info['email'],
                        "Transaction Alert: Withdrawal",
                        f"Hello {user_info['full_name']},\n\nYour account {session['acc_num']} has been debited by ₹{amount:,.2f} for a withdrawal.\n\nTransaction ID: {tran_id}\nDate: {date.today()}"
                    )
                    
                flash('Withdrawal successful!', 'success')
            except mysql.connector.Error as err:
                conn.rollback()
                flash(f'Error: {err}', 'danger')
        else:
            flash('Insufficient balance.', 'warning')
        
        cursor.close()
        conn.close()
        return redirect(url_for('dashboard'))
        
    return render_template('withdraw.html')

@app.route('/deposit', methods=['GET', 'POST'])
def deposit():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        amount = float(request.form['amount'])
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("UPDATE accounts SET balance = balance + %s WHERE acc_num = %s", (amount, session['acc_num']))
            tran_id = generate_tran_id()
            query = "INSERT INTO history (tran_id, from_acc, to_acc, date, amount, type) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (tran_id, 'Self', session['acc_num'], date.today(), amount, 'Deposit'))
            conn.commit()
            
            # Send Email
            user_info = get_user_email(session['acc_num'])
            if user_info:
                send_transaction_email(
                    user_info['email'],
                    "Transaction Alert: Deposit",
                    f"Hello {user_info['full_name']},\n\nYour account {session['acc_num']} has been credited with ₹{amount:,.2f} for a deposit.\n\nTransaction ID: {tran_id}\nDate: {date.today()}"
                )
                
            flash('Deposit successful!', 'success')
        except mysql.connector.Error as err:
            conn.rollback()
            flash(f'Error: {err}', 'danger')
        finally:
            cursor.close()
            conn.close()
            
        return redirect(url_for('dashboard'))
        
    return render_template('deposit.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM history WHERE from_acc = %s OR to_acc = %s ORDER BY date DESC"
    cursor.execute(query, (session['acc_num'], session['acc_num']))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('history.html', transactions=transactions)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

# Admin Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username.lower() == 'admin' and password.lower() == 'admin':
            session['admin_logged_in'] = True
            flash('Welcome Admin!', 'success')
            return redirect(url_for('admin_users'))
        else:
            flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/users')
def admin_users():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM accounts")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('admin_users.html', users=users)

@app.route('/admin/delete_user/<int:id>')
def delete_user(id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get user_id first to delete biometrics
    cursor.execute("SELECT user_id FROM accounts WHERE id = %s", (id,))
    user = cursor.fetchone()
    
    if user:
        auth.delete_user(user['user_id'])
        cursor.execute("DELETE FROM accounts WHERE id = %s", (id,))
        conn.commit()
        flash('User and biometric data deleted successfully.', 'success')
    else:
        flash('User not found.', 'danger')
        
    cursor.close()
    conn.close()
    return redirect(url_for('admin_users'))

@app.route('/admin/banks')
def admin_banks():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM banks")
    banks = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('admin_banks.html', banks=banks)

@app.route('/admin/add_bank', methods=['POST'])
def add_bank():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    bank_name = request.form['bank_name']
    branch_name = request.form['branch_name']
    ifsc = request.form['ifsc']
    address = request.form['address']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "INSERT INTO banks (bank_name, branch_name, ifsc, address) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (bank_name, branch_name, ifsc, address))
        conn.commit()
        flash('Bank added successfully!', 'success')
    except mysql.connector.Error as err:
        flash(f'Error adding bank: {err}', 'danger')
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('admin_banks'))

@app.route('/admin/delete_bank/<int:id>')
def delete_bank(id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 1. Get bank name to identify users
        cursor.execute("SELECT bank_name FROM banks WHERE id = %s", (id,))
        bank = cursor.fetchone()
        
        if bank:
            bank_name = bank['bank_name']
            
            # 2. Find all users of this bank and delete their biometrics
            cursor.execute("SELECT user_id FROM accounts WHERE bank_name = %s", (bank_name,))
            users = cursor.fetchall()
            for u in users:
                auth.delete_user(u['user_id'])
            
            # 3. Delete users from database (manual cascading)
            cursor.execute("DELETE FROM accounts WHERE bank_name = %s", (bank_name,))
            
            # 4. Delete the bank itself
            cursor.execute("DELETE FROM banks WHERE id = %s", (id,))
            
            conn.commit()
            flash(f'Bank "{bank_name}" and all associated user accounts deleted.', 'success')
        else:
            flash('Bank not found.', 'danger')
            
    except mysql.connector.Error as err:
        conn.rollback()
        flash(f'Error deleting bank: {err}', 'danger')
    finally:
        cursor.close()
        conn.close()
        
    return redirect(url_for('admin_banks'))

if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
