"""Synthetic business database builder.

Generates a SQLite database at data/business.db with realistic e-commerce
and sales data including customers, products, categories, suppliers, orders,
order_items, employees, and payments.
"""

from __future__ import annotations

import datetime
import os
import random
import sqlite3
from pathlib import Path


def get_db_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    return base_dir / "business.db"


def build_database(db_path: Path | str | None = None, seed: int = 42) -> Path:
    """Builds synthetic SQLite database deterministically using a fixed random seed."""
    if db_path is None:
        target_path = get_db_path()
    else:
        target_path = Path(db_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        target_path.unlink()

    random.seed(seed)

    conn = sqlite3.connect(str(target_path))
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Categories
    cursor.execute(
        """
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        """
    )

    # 2. Suppliers
    cursor.execute(
        """
        CREATE TABLE suppliers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            country TEXT NOT NULL
        );
        """
    )

    # 3. Employees
    cursor.execute(
        """
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            hire_date TEXT NOT NULL,
            manager_id INTEGER,
            region TEXT NOT NULL,
            FOREIGN KEY (manager_id) REFERENCES employees(id)
        );
        """
    )

    # 4. Products
    cursor.execute(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            cost REAL NOT NULL,
            stock_quantity INTEGER NOT NULL,
            supplier_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id),
            FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
        );
        """
    )

    # 5. Customers
    cursor.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            country TEXT NOT NULL,
            signup_date TEXT NOT NULL,
            segment TEXT NOT NULL
        );
        """
    )

    # 6. Orders
    cursor.execute(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            employee_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            ship_date TEXT,
            status TEXT NOT NULL,
            shipping_city TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        );
        """
    )

    # 7. Order Items
    cursor.execute(
        """
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            discount REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        """
    )

    # 8. Payments
    cursor.execute(
        """
        CREATE TABLE payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            payment_date TEXT NOT NULL,
            method TEXT NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );
        """
    )

    # Populate Categories
    categories = [
        "Laptops & Computers",
        "Computer Accessories",
        "Monitors & Displays",
        "Networking & Servers",
        "Office Furniture",
        "Audio & Headsets",
        "Storage & Backup",
        "Printers & Supplies",
    ]
    cursor.executemany("INSERT INTO categories (name) VALUES (?)", [(c,) for c in categories])

    # Populate Suppliers
    suppliers = [
        ("TechSupply Global", "United States"),
        ("Nordic Component AG", "Germany"),
        ("Tokyo Micro Systems", "Japan"),
        ("Shenzhen Silicon Ltd", "China"),
        ("Bengaluru Tech Hub", "India"),
        ("Pacific Peripheral Co", "Taiwan"),
        ("EuroLogistics B.V.", "Netherlands"),
        ("Maple Hardware Inc", "Canada"),
        ("Thames Electronics", "United Kingdom"),
        ("Seoul Digital Works", "South Korea"),
    ]
    cursor.executemany("INSERT INTO suppliers (name, country) VALUES (?, ?)", suppliers)

    # Populate Employees (with hierarchical manager_id)
    # 1 CEO, 3 Regional Managers, 16 Representatives
    employees_data = [
        ("Eleanor Vance", "VP of Sales", "2021-01-15", None, "Global"),
        ("Marcus Brody", "Regional Sales Manager", "2021-03-01", 1, "North America"),
        ("Sofia Rodriguez", "Regional Sales Manager", "2021-04-10", 1, "Europe"),
        ("Kenji Sato", "Regional Sales Manager", "2021-05-20", 1, "Asia-Pacific"),
        ("Aarav Patel", "Regional Sales Manager", "2021-06-01", 1, "South Asia"),
    ]
    cursor.executemany(
        "INSERT INTO employees (name, role, hire_date, manager_id, region) VALUES (?, ?, ?, ?, ?)",
        employees_data,
    )

    rep_names = [
        ("Alice Johnson", 2, "North America"),
        ("Bob Smith", 2, "North America"),
        ("Carol White", 2, "North America"),
        ("David Miller", 2, "North America"),
        ("Emma Davis", 3, "Europe"),
        ("Fabian Schmidt", 3, "Europe"),
        ("Greta Weber", 3, "Europe"),
        ("Hans Mueller", 3, "Europe"),
        ("Hina Takahashi", 4, "Asia-Pacific"),
        ("Ichiro Tanaka", 4, "Asia-Pacific"),
        ("Jin-Woo Park", 4, "Asia-Pacific"),
        ("Mei-Ling Chen", 4, "Asia-Pacific"),
        ("Priya Sharma", 5, "South Asia"),
        ("Rohan Gupta", 5, "South Asia"),
        ("Ananya Iyer", 5, "South Asia"),
    ]
    hire_base = datetime.date(2022, 1, 1)
    rep_records = []
    for name, mgr, reg in rep_names:
        h_date = hire_base + datetime.timedelta(days=random.randint(0, 400))
        rep_records.append((name, "Sales Representative", h_date.isoformat(), mgr, reg))
    cursor.executemany(
        "INSERT INTO employees (name, role, hire_date, manager_id, region) VALUES (?, ?, ?, ?, ?)",
        rep_records,
    )

    # Populate Products (80 products)
    products_list = [
        # Laptops & Computers (cat 1)
        ("Apex ProBook 15", 1, 1299.99, 850.0, 45, 1),
        ("Apex Air Ultra 13", 1, 999.50, 650.0, 60, 1),
        ("Apex Workstation Tower", 1, 2199.00, 1450.0, 20, 2),
        ("NanoMini Desktop PC", 1, 549.00, 360.0, 75, 4),
        ("Zenith Creator Studio 16", 1, 1899.99, 1250.0, 15, 6),
        ("ChronoBook Enterprise", 1, 1150.00, 780.0, 40, 8),
        ("CyberShield Rugged Laptop", 1, 2450.00, 1600.0, 8, 2),
        ("OmniBook Lite", 1, 429.00, 290.0, 90, 4),
        ("Apex Gaming Station X", 1, 2799.00, 1900.0, 12, 10),
        ("EcoDesktop Mini", 1, 480.00, 320.0, 30, 7),
        # Computer Accessories (cat 2)
        ("ErgoClick Wireless Mouse", 2, 49.99, 22.0, 220, 3),
        ("ProMechanical RGB Keyboard", 2, 129.99, 65.0, 140, 6),
        ("SilentKey Low-Profile Keyboard", 2, 69.99, 32.0, 180, 4),
        ("Precision Trackball Pro", 2, 89.50, 42.0, 50, 3),
        ("Dual 4K USB-C Hub Dock", 2, 159.00, 75.0, 95, 6),
        ("UltraSpeed 100W GaN Charger", 2, 45.00, 18.0, 310, 4),
        ("Braided Thunderbolt 4 Cable 2m", 2, 34.99, 11.0, 260, 4),
        ("Ergonomic Gel Wrist Rest", 2, 19.99, 6.5, 400, 7),
        ("Executive Leather Desk Mat", 2, 39.00, 14.0, 175, 7),
        ("Aluminum Laptop Cooling Stand", 2, 54.00, 24.0, 110, 6),
        # Monitors & Displays (cat 3)
        ("UltraView 27-inch 4K IPS", 3, 389.00, 240.0, 65, 3),
        ("UltraView 34-inch Curved Ultrawide", 3, 649.99, 410.0, 35, 10),
        ("ClearVue 24-inch FHD Office Monitor", 3, 169.99, 105.0, 150, 8),
        ("ProColor 32-inch HDR Designer", 3, 899.00, 580.0, 22, 3),
        ("SpeedSync 27-inch 165Hz Gaming", 3, 299.99, 185.0, 80, 10),
        ("PortaView 15.6-inch USB-C Monitor", 3, 199.00, 120.0, 45, 6),
        ("UltraView 49-inch Super Ultrawide", 3, 1199.00, 790.0, 10, 10),
        ("Dual-Arm Gas Spring Monitor Mount", 3, 79.99, 32.0, 130, 8),
        ("Privacy Filter Screen 27-inch", 3, 44.50, 15.0, 85, 7),
        ("Monitor Light Bar Smart Dimming", 3, 59.99, 25.0, 160, 4),
        # Networking & Servers (cat 4)
        ("NetCore WiFi 6E Mesh Router (3-Pack)", 4, 329.00, 195.0, 40, 1),
        ("Gigabit 24-Port Managed PoE Switch", 4, 450.00, 270.0, 25, 2),
        ("Enterprise Dual-Band Access Point", 4, 189.00, 110.0, 70, 1),
        ("Hardware Firewall Appliance 1Gbps", 4, 699.00, 430.0, 18, 2),
        ("Cat6A Shielded Ethernet Cable 50ft", 4, 24.99, 8.5, 350, 4),
        ("8-Port Desktop Unmanaged Switch", 4, 39.99, 17.0, 210, 4),
        ("SFP+ 10Gb Transceiver Module", 4, 55.00, 20.0, 180, 6),
        ("Rackmount 1U PDU 8-Outlets", 4, 89.00, 42.0, 60, 2),
        ("Network Patch Panel 24-Port", 4, 48.00, 21.0, 95, 2),
        ("Industrial 4G/LTE Failover Modem", 4, 279.00, 165.0, 30, 1),
        # Office Furniture (cat 5)
        ("AeroErgo Mesh Task Chair", 5, 349.00, 180.0, 50, 7),
        ("Apex Electric Standing Desk 60x30", 5, 599.00, 340.0, 30, 8),
        ("Executive Leather High-Back Chair", 5, 499.00, 260.0, 20, 7),
        ("Under-Desk Steel Mobile Pedestal", 5, 149.00, 70.0, 45, 8),
        ("Anti-Fatigue Standing Mat", 5, 49.99, 18.0, 120, 7),
        ("Acoustic Desk Partition Screen", 5, 89.00, 35.0, 65, 8),
        ("Conference Table 8-Person Oak", 5, 1250.00, 720.0, 6, 8),
        ("Steel Wire Cable Management Spine", 5, 29.99, 10.0, 190, 8),
        ("Compact Active Stool", 5, 119.00, 55.0, 40, 7),
        ("Footrest with Heating Massage", 5, 69.00, 28.0, 75, 7),
        # Audio & Headsets (cat 6)
        ("ClearVoice ANC Wireless Headset", 6, 179.99, 85.0, 130, 9),
        ("StudioMaster USB Cardioid Mic", 6, 119.00, 52.0, 90, 9),
        ("Conference 360 Speakerphone", 6, 229.00, 115.0, 45, 1),
        ("TrueWireless Earbuds Pro ANC", 6, 149.50, 65.0, 160, 9),
        ("Boom Mic Broadcast Arm", 6, 49.00, 19.0, 110, 9),
        ("Desktop Stereo Soundbar Hi-Res", 6, 89.99, 38.0, 85, 9),
        ("CallCenter Single-Ear Wired Headset", 6, 39.99, 15.0, 250, 9),
        ("Pop Filter & Shockmount Kit", 6, 29.99, 9.5, 140, 9),
        ("Acoustic Foam Soundproofing (12-Pack)", 6, 35.00, 12.0, 170, 7),
        ("Bluetooth 5.3 Audio Transmitter", 6, 32.50, 12.5, 95, 4),
        # Storage & Backup (cat 7)
        ("FastVault 2TB NVMe Portable SSD", 7, 169.99, 95.0, 140, 10),
        ("FastVault 4TB Rugged External HDD", 7, 119.00, 68.0, 110, 10),
        ("ServerGrade 16TB Enterprise SATA", 7, 389.00, 230.0, 35, 10),
        ("NAS Dual-Bay Home/Office Enclosure", 7, 249.00, 145.0, 40, 6),
        ("SecureKey 128GB Hardware Encrypted", 7, 69.99, 28.0, 150, 6),
        ("PCIe Gen4 2TB Internal M.2 SSD", 7, 149.00, 80.0, 125, 10),
        ("External Blu-Ray Writer Drive", 7, 95.00, 48.0, 50, 3),
        ("High-Endurance 256GB MicroSDXC", 7, 39.99, 15.0, 220, 6),
        ("Hard Drive Docking Station Dual Bay", 7, 59.99, 26.0, 70, 4),
        ("NAS 4-Bay Rackmount System", 7, 799.00, 490.0, 12, 6),
        # Printers & Supplies (cat 8)
        ("OfficeJet Pro Color All-in-One", 8, 299.99, 180.0, 45, 3),
        ("LaserJet Enterprise Monochrome Duplex", 8, 489.00, 310.0, 30, 3),
        ("High-Yield Black Toner Cartridge", 8, 89.99, 32.0, 210, 3),
        ("Color Toner Multipack (CMY)", 8, 199.99, 75.0, 115, 3),
        ("Multipurpose Copy Paper 20lb (5 Reams)", 8, 38.00, 16.0, 320, 8),
        ("Heavy-Duty Thermal Label Printer", 8, 179.00, 92.0, 65, 4),
        ("Direct Thermal Shipping Labels (6 Rolls)", 8, 28.50, 9.0, 280, 4),
        ("Sheetfed Desktop Document Scanner", 8, 369.00, 225.0, 25, 3),
        ("3D Rapid Prototyping Printer", 8, 649.00, 390.0, 15, 6),
        ("PLA 3D Printing Filament 1kg", 8, 22.99, 8.0, 190, 4),
    ]
    cursor.executemany(
        """
        INSERT INTO products (name, category_id, unit_price, cost, stock_quantity, supplier_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        products_list,
    )

    # Populate Customers (300 customers)
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
        "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
        "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
        "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen", "Stephen", "Anna",
        "Larry", "Brenda", "Justin", "Pamela", "Scott", "Nicole", "Brandon", "Emma",
        "Benjamin", "Samantha", "Samuel", "Katherine", "Gregory", "Christine", "Frank", "Debra",
        "Alexander", "Rachel", "Raymond", "Catherine", "Patrick", "Carolyn", "Jack", "Janet",
        "Dennis", "Ruth", "Jerry", "Maria", "Tyler", "Heather", "Aaron", "Diane",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
        "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
        "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
        "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
        "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
        "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
        "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    ]
    cities_states = [
        ("New York", "NY", "United States"),
        ("Los Angeles", "CA", "United States"),
        ("Chicago", "IL", "United States"),
        ("Houston", "TX", "United States"),
        ("Phoenix", "AZ", "United States"),
        ("Philadelphia", "PA", "United States"),
        ("San Antonio", "TX", "United States"),
        ("San Diego", "CA", "United States"),
        ("Dallas", "TX", "United States"),
        ("Austin", "TX", "United States"),
        ("Seattle", "WA", "United States"),
        ("Denver", "CO", "United States"),
        ("Boston", "MA", "United States"),
        ("Atlanta", "GA", "United States"),
        ("Miami", "FL", "United States"),
        ("Toronto", "ON", "Canada"),
        ("Vancouver", "BC", "Canada"),
        ("Montreal", "QC", "Canada"),
        ("London", "ENG", "United Kingdom"),
        ("Manchester", "ENG", "United Kingdom"),
        ("Berlin", "BE", "Germany"),
        ("Munich", "BY", "Germany"),
        ("Mumbai", "MH", "India"),
        ("Bengaluru", "KA", "India"),
        ("Sydney", "NSW", "Australia"),
    ]
    segments = ["Consumer", "Corporate", "Small Business"]
    segment_weights = [0.5, 0.3, 0.2]

    customers_data = []
    used_emails = set()
    base_signup = datetime.date(2022, 1, 1)

    for i in range(1, 301):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        full_name = f"{fn} {ln}"
        email_prefix = f"{fn.lower()}.{ln.lower()}{i}"
        domain = random.choice(["gmail.com", "outlook.com", "enterprise.net", "techcorp.io", "consulting.org"])
        email = f"{email_prefix}@{domain}"
        city, state, country = random.choice(cities_states)
        signup_dt = base_signup + datetime.timedelta(days=random.randint(0, 750))
        segment = random.choices(segments, weights=segment_weights)[0]
        customers_data.append((full_name, email, city, state, country, signup_dt.isoformat(), segment))

    cursor.executemany(
        """
        INSERT INTO customers (name, email, city, state, country, signup_date, segment)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        customers_data,
    )

    # Edge case: Customers 281-300 have NO orders (never ordered)
    # Edge case: Products 78, 79, 80 are NEVER ordered (stock = 15, 190, etc.)
    eligible_customers = list(range(1, 281))
    eligible_products = list(range(1, 78))  # 1 to 77

    # Populate Orders (3000 orders) spanning 24+ months (2024-01-01 to 2025-12-31)
    # Status distribution: Completed (70%), Shipped (15%), Processing (8%), Cancelled (5%), Refunded (2%)
    order_statuses = ["Completed", "Shipped", "Processing", "Cancelled", "Refunded"]
    status_weights = [0.70, 0.15, 0.08, 0.05, 0.02]

    # Map customer to their city for shipping_city
    cursor.execute("SELECT id, city FROM customers")
    cust_city_map = dict(cursor.fetchall())

    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    total_days = (end_date - start_date).days

    orders_records = []
    order_status_map = {}
    order_date_map = {}

    for o_id in range(1, 3001):
        cust_id = random.choice(eligible_customers)
        emp_id = random.randint(1, len(employees_data) + len(rep_names))
        # date distribution
        day_offset = random.randint(0, total_days)
        o_date = start_date + datetime.timedelta(days=day_offset)
        status = random.choices(order_statuses, weights=status_weights)[0]

        if status in ("Processing", "Cancelled"):
            ship_date = None
        else:
            ship_date = (o_date + datetime.timedelta(days=random.randint(1, 6))).isoformat()

        shipping_city = cust_city_map[cust_id]
        orders_records.append((cust_id, emp_id, o_date.isoformat(), ship_date, status, shipping_city))
        order_status_map[o_id] = status
        order_date_map[o_id] = o_date

    cursor.executemany(
        """
        INSERT INTO orders (customer_id, employee_id, order_date, ship_date, status, shipping_city)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        orders_records,
    )

    # Populate Order Items (roughly 9000 items, 1 to 5 per order)
    cursor.execute("SELECT id, unit_price FROM products")
    product_price_map = dict(cursor.fetchall())

    order_items_records = []
    order_totals = {o_id: 0.0 for o_id in range(1, 3001)}

    for o_id in range(1, 3001):
        # 1 to 5 distinct items per order (average ~3 items per order = ~9000 items)
        num_items = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.25, 0.35, 0.20, 0.10])[0]
        selected_prods = random.sample(eligible_products, num_items)
        for p_id in selected_prods:
            qty = random.choices([1, 2, 3, 4, 5, 8, 10], weights=[0.5, 0.25, 0.12, 0.06, 0.04, 0.02, 0.01])[0]
            base_price = product_price_map[p_id]
            discount = random.choices([0.0, 0.05, 0.10, 0.15, 0.20], weights=[0.70, 0.15, 0.08, 0.05, 0.02])[0]
            line_total = round(qty * base_price * (1.0 - discount), 2)
            order_totals[o_id] += line_total
            order_items_records.append((o_id, p_id, qty, base_price, discount))

    cursor.executemany(
        """
        INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
        VALUES (?, ?, ?, ?, ?)
        """,
        order_items_records,
    )

    # Populate Payments
    # Processing and Cancelled orders might not have payments
    payment_methods = ["Credit Card", "PayPal", "Bank Transfer", "Debit Card"]
    method_weights = [0.55, 0.25, 0.12, 0.08]
    payments_records = []

    for o_id in range(1, 3001):
        status = order_status_map[o_id]
        if status == "Cancelled" and random.random() < 0.8:
            # 80% cancelled orders have no payment record
            continue
        if status == "Processing" and random.random() < 0.25:
            # 25% processing orders are unpaid
            continue

        total_amount = round(order_totals[o_id], 2)
        if total_amount <= 0:
            total_amount = 25.0

        p_date = (order_date_map[o_id] + datetime.timedelta(hours=random.randint(0, 48))).isoformat()
        method = random.choices(payment_methods, weights=method_weights)[0]
        payments_records.append((o_id, p_date, method, total_amount))

    cursor.executemany(
        """
        INSERT INTO payments (order_id, payment_date, method, amount)
        VALUES (?, ?, ?, ?)
        """,
        payments_records,
    )

    conn.commit()
    conn.close()

    return target_path


def print_database_summary(db_path: Path | str | None = None) -> None:
    """Prints table row counts and summary statistics."""
    if db_path is None:
        target_path = get_db_path()
    else:
        target_path = Path(db_path)

    conn = sqlite3.connect(str(target_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]

    print("=" * 50)
    print(f"DATABASE SUMMARY: {target_path}")
    print("=" * 50)
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:<15}: {count:>6} rows")
    print("=" * 50)

    conn.close()


if __name__ == "__main__":
    path = build_database()
    print_database_summary(path)
