import asyncio
import os
import sys
import json
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Fix for Windows asyncio and psycopg compatibility
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Try to find .env in root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(root_dir, ".env")
load_dotenv(env_path)

# Build connection string matching main.py logic
conn_info = f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')}"

async def get_conn():
    if not os.getenv('DB_HOST'):
        print(f"Error: Database configuration (DB_HOST, etc.) not found. Searched in: {env_path}")
        sys.exit(1)
    return await psycopg.AsyncConnection.connect(conn_info, autocommit=True)

async def promote_user(email: str):
    async with await get_conn() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, email, role FROM users WHERE email = %s", (email.lower(),))
            user = await cur.fetchone()
            
            if not user:
                print(f"Error: User with email '{email}' not found.")
                return
            
            if user["role"] == "admin":
                print(f"User '{email}' is already an admin.")
            else:
                await cur.execute("UPDATE users SET role = 'admin' WHERE id = %s", (user["id"],))
                # Need to commit if not in autocommit mode, but psycopg connections usually need explicit commit or use TRANSACTION block
                # However, within 'async with conn', it usually manages transaction.
                print(f"Successfully promoted '{email}' to admin.")

async def list_users():
    async with await get_conn() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, email, role, name FROM users")
            users = await cur.fetchall()
            print(f"{'ID':<5} {'Email':<30} {'Role':<10} {'Name':<20}")
            print("-" * 75)
            for u in users:
                print(f"{u['id']:<5} {u['email']:<30} {u['role']:<10} {u['name'] or 'N/A':<20}")

async def toggle_registration(allow: bool):
    async with await get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO system_settings (key, value) VALUES ('registration', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (json.dumps({"allow": allow}),))
            state = "enabled" if allow else "disabled"
            print(f"Registration has been {state}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python admin_cli.py list              - List all users")
        print("  python admin_cli.py promote <email>   - Promote user to admin")
        print("  python admin_cli.py reg on|off        - Toggle registration")
        sys.exit(1)

    command = sys.argv[1].lower()
    
    if command == "list":
        asyncio.run(list_users())
    elif command == "promote" and len(sys.argv) > 2:
        asyncio.run(promote_user(sys.argv[2]))
    elif command == "reg" and len(sys.argv) > 2:
        arg = sys.argv[2].lower()
        if arg in ["on", "true", "1"]:
            asyncio.run(toggle_registration(True))
        elif arg in ["off", "false", "0"]:
            asyncio.run(toggle_registration(False))
        else:
            print("Invalid argument for reg. Use on/off.")
    else:
        print("Unknown command or missing arguments.")
