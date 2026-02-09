import os
import json
import logging
import asyncio
import sys
import time
import base64
import hashlib
import hmac
import secrets
import uuid
import mimetypes
import shutil
import glob
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

# 解决 Windows 下 Psycopg 异步连接池与默认事件循环 (ProactorEventLoop) 的兼容性问题
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import httpx
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Query, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv

# --- 1. 配置与日志模块 ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("InsightVault")

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
ACCESS_TOKEN_TTL_SECONDS = 60 * 60 * 24
PASSWORD_ITERATIONS = 120_000
CLOUD_STORAGE_DIR = os.path.abspath(os.getenv("CLOUD_STORAGE_DIR", "storage"))
OCR_MODEL = os.getenv("OCR_MODEL", "gpt-4o-mini").strip()

def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)

def _jwt_sign(message: str, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), message.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(sig)

def create_access_token(user_id: int, role: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": int(time.time()) + ACCESS_TOKEN_TTL_SECONDS
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}"
    signature = _jwt_sign(signing_input, JWT_SECRET)
    return f"{signing_input}.{signature}"

def decode_access_token(token: str) -> dict:
    try:
        header_b64, payload_b64, signature = token.split(".")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token format")

    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = _jwt_sign(signing_input, JWT_SECRET)
    if not hmac.compare_digest(signature, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    payload_raw = _b64url_decode(payload_b64)
    payload = json.loads(payload_raw)
    if payload.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload

def hash_password(password: str, salt: Optional[str] = None) -> dict:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS
    )
    return {"salt": salt, "hash": digest.hex()}

def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS
    ).hex()
    return hmac.compare_digest(digest, expected_hash)

# 耗时统计工具
class Profiler:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = (self.end - self.start) * 1000 # 转为毫秒

# --- 2. 数据库连接池初始化 ---

# 构造连接字符串
conn_info = f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化连接池
    try:
        if not JWT_SECRET:
            logger.error("JWT_SECRET 未配置，无法启动鉴权系统")
            raise RuntimeError("JWT_SECRET is required")

        # 显式使用 autocommit，并添加心跳检查逻辑
        app.state.db_pool = AsyncConnectionPool(
            conninfo=conn_info, 
            min_size=1, 
            max_size=10, 
            open=False,
            kwargs={"autocommit": True}
        )
        await app.state.db_pool.open()
        await app.state.db_pool.wait() # 确保至少有一个连接可用
        logger.info("成功初始化异步数据库连接池 (Max: 10)")

        os.makedirs(CLOUD_STORAGE_DIR, exist_ok=True)

        # --- 动态维度检测逻辑 ---
        logger.info("正在检测模型向量维度...")
        test_vec = await get_embedding("DimCheck")
        detected_dim = len(test_vec)
        logger.info(f"检测到模型维度为: {detected_dim}")
        
        # 初始化数据库并同步维度
        async with app.state.db_pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL
                );
            """)
            # Initialize default settings
            await conn.execute("""
                INSERT INTO system_settings (key, value)
                VALUES ('registration', '{"allow": true}'::jsonb)
                ON CONFLICT (key) DO NOTHING;
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS groups (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_user_id INTEGER REFERENCES users(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS group_members (
                    user_id INTEGER REFERENCES users(id),
                    group_id INTEGER REFERENCES groups(id),
                    role TEXT NOT NULL DEFAULT 'member',
                    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, group_id)
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS group_invites (
                    code TEXT PRIMARY KEY,
                    group_id INTEGER REFERENCES groups(id),
                    created_by INTEGER REFERENCES users(id),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    max_uses INTEGER DEFAULT 20,
                    uses INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_user_id INTEGER REFERENCES users(id),
                    visibility TEXT NOT NULL DEFAULT 'private',
                    group_id INTEGER REFERENCES groups(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS item_tags (
                    item_id INTEGER REFERENCES intelligence_vault(id) ON DELETE CASCADE,
                    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                    PRIMARY KEY (item_id, tag_id)
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_api_keys (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    key_name TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    scopes TEXT NOT NULL DEFAULT 'read,write',
                    last_used TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user ON user_api_keys (user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON user_api_keys (key_hash);")
            await conn.execute("ALTER TABLE user_api_keys ADD COLUMN IF NOT EXISTS scopes TEXT NOT NULL DEFAULT 'read,write';")
            await conn.execute("ALTER TABLE user_api_keys ALTER COLUMN scopes SET DEFAULT 'read,write,reading,cloud';")
            
            # 强力同步数据库维度
            try:
                # 检查表是否存在
                async with conn.cursor() as cur:
                    await cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'intelligence_vault' AND column_name = 'embedding';")
                    if await cur.fetchone():
                        # 获取当前数据库字段维度
                        await cur.execute("""
                            SELECT atttypmod 
                            FROM pg_attribute 
                            WHERE attrelid = 'intelligence_vault'::regclass 
                            AND attname = 'embedding';
                        """)
                        row = await cur.fetchone()
                        current_db_dim = row[0] if row else -1
                        
                        if current_db_dim != detected_dim:
                            logger.info(f"维度不匹配 (现有: {current_db_dim}, 模型: {detected_dim})，正在调整数据库...")
                            await conn.execute("DROP INDEX IF EXISTS idx_iv_hnsw;")
                            await conn.execute(f"ALTER TABLE intelligence_vault ALTER COLUMN embedding TYPE vector({detected_dim}) USING NULL;")
                            logger.info(f"数据库维度已强制调整为 {detected_dim}")
            except Exception as e:
                logger.debug(f"维度检查流转: {e}")

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS intelligence_vault (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT NOT NULL,
                    embedding vector({detected_dim}),
                    owner_user_id INTEGER REFERENCES users(id),
                    visibility TEXT DEFAULT 'private',
                    group_id INTEGER REFERENCES groups(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    vectorization_status VARCHAR(20) DEFAULT 'pending'
                );
            """)
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS title TEXT;")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS owner_user_id INTEGER REFERENCES users(id);")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS visibility TEXT DEFAULT 'private';")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS group_id INTEGER REFERENCES groups(id);")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS vectorization_status VARCHAR(20) DEFAULT 'pending';")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE;")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS is_favorited BOOLEAN DEFAULT FALSE;")
            await conn.execute("UPDATE intelligence_vault SET visibility = 'private' WHERE visibility IS NULL;")
            # 修正状态迁移逻辑：如果有向量但状态还是 pending，则改为 success
            await conn.execute("UPDATE intelligence_vault SET vectorization_status = 'success' WHERE embedding IS NOT NULL AND (vectorization_status IS NULL OR vectorization_status = 'pending');")
            await conn.execute("UPDATE intelligence_vault SET vectorization_status = 'pending' WHERE embedding IS NULL AND vectorization_status IS NULL;")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_iv_owner ON intelligence_vault (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_iv_group ON intelligence_vault (group_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_iv_hnsw ON intelligence_vault USING hnsw (embedding vector_cosine_ops);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_owner ON tags (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_group ON tags (group_id);")
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_tags_private ON tags (owner_user_id, name) WHERE visibility = 'private';")
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_tags_group ON tags (group_id, name) WHERE visibility = 'group';")
            
            # 阅读列表表
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS reading_list (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    content TEXT,
                    source VARCHAR(255),
                    cover_image TEXT,
                    has_content BOOLEAN DEFAULT FALSE,
                    owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    is_read BOOLEAN DEFAULT FALSE,
                    is_archived BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("ALTER TABLE reading_list ADD COLUMN IF NOT EXISTS has_content BOOLEAN DEFAULT FALSE;")
            await conn.execute("ALTER TABLE reading_list ADD COLUMN IF NOT EXISTS is_read BOOLEAN DEFAULT FALSE;")
            await conn.execute("ALTER TABLE reading_list ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE;")
            await conn.execute("ALTER TABLE reading_list ADD COLUMN IF NOT EXISTS is_favorited BOOLEAN DEFAULT FALSE;")
            await conn.execute("ALTER TABLE reading_list ADD COLUMN IF NOT EXISTS tags TEXT[];")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_owner ON reading_list (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_created ON reading_list (created_at DESC);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_status ON reading_list (owner_user_id, is_read, is_archived);")

            # 云盘表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cloud_items (
                    id SERIAL PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    parent_id INTEGER REFERENCES cloud_items(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    item_type TEXT NOT NULL CHECK (item_type IN ('file', 'folder')),
                    size BIGINT DEFAULT 0,
                    mime_type TEXT,
                    storage_path TEXT,
                    checksum TEXT,
                    visibility TEXT DEFAULT 'private',
                    group_id INTEGER REFERENCES groups(id),
                    is_shared BOOLEAN DEFAULT FALSE,
                    is_archived BOOLEAN DEFAULT FALSE,
                    is_favorited BOOLEAN DEFAULT FALSE,
                    tags TEXT[],
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cloud_versions (
                    id SERIAL PRIMARY KEY,
                    item_id INTEGER REFERENCES cloud_items(id) ON DELETE CASCADE,
                    version_num INTEGER NOT NULL,
                    size BIGINT,
                    checksum TEXT,
                    storage_path TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER REFERENCES users(id)
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cloud_shares (
                    token TEXT PRIMARY KEY,
                    item_id INTEGER REFERENCES cloud_items(id) ON DELETE CASCADE,
                    created_by INTEGER REFERENCES users(id),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    max_uses INTEGER DEFAULT 0,
                    uses INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_owner ON cloud_items (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_parent ON cloud_items (parent_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_group ON cloud_items (group_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_name ON cloud_items (name);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_shared ON cloud_items (is_shared);")

            # 小程序 / 快船相关表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS mini_user_settings (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    auto_provide BOOLEAN DEFAULT FALSE,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS mini_data_requests (
                    id SERIAL PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    app_name TEXT NOT NULL,
                    data_types TEXT[],
                    purpose TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    decision_note TEXT,
                    grant_token TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    decided_at TIMESTAMP WITH TIME ZONE
                );
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_mini_requests_owner ON mini_data_requests (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_mini_requests_status ON mini_data_requests (status);")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS mini_upload_sessions (
                    code TEXT PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP WITH TIME ZONE
                );
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_mini_sessions_owner ON mini_upload_sessions (owner_user_id);")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS mini_upload_files (
                    id SERIAL PRIMARY KEY,
                    session_code TEXT REFERENCES mini_upload_sessions(code) ON DELETE CASCADE,
                    cloud_item_id INTEGER REFERENCES cloud_items(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    size BIGINT DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_mini_files_session ON mini_upload_files (session_code);")
            
            logger.info("数据库环境检查完成")

            # Bootstrap admin if configured
            bootstrap_email = os.getenv("ADMIN_BOOTSTRAP_EMAIL", "").strip()
            bootstrap_password = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "").strip()
            if bootstrap_email and bootstrap_password:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT id FROM users WHERE email = %s", (bootstrap_email,))
                    if not await cur.fetchone():
                        pw = hash_password(bootstrap_password)
                        await cur.execute(
                            "INSERT INTO users (email, name, password_hash, password_salt, role) VALUES (%s, %s, %s, %s, 'admin')",
                            (bootstrap_email, "Admin", pw["hash"], pw["salt"])
                        )
                        logger.info("已创建引导管理员账号")
    except Exception as e:
        logger.error(f"生命周期启动失败: {e}")
        raise e
    
    yield
    # 关闭时：关闭连接池
    await app.state.db_pool.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- 3. 核心业务逻辑 ---

# 简单的异步 LRU 缓存实现
_embedding_cache = {}

async def get_embedding(text: str):
    # 使用缓存
    if text in _embedding_cache:
        return _embedding_cache[text]
    
    base_url = os.getenv("AI_BASE_URL").rstrip('/')
    endpoint = f"{base_url}/embeddings"
    headers = {"Authorization": f"Bearer {os.getenv('AI_API_KEY')}"}
    payload = {"model": os.getenv("EMBEDDING_MODEL"), "input": text}
    
    with Profiler() as p:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            vec = response.json()['data'][0]['embedding']
    
    logger.info(f"Embedding API 请求完成 | 耗时: {p.duration:.2f}ms | 内容: {text[:10]}...")
    
    # 维护简单缓存
    if len(_embedding_cache) > 128:
        _embedding_cache.pop(next(iter(_embedding_cache)))
    _embedding_cache[text] = vec
    return vec

async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    token = None
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
    else:
        # 允许从查询参数中获取 token，用于下载等无法设置 Header 的场景
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
        
    payload = decode_access_token(token)
    user_id = int(payload.get("sub", 0))

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, email, name, role FROM users WHERE id = %s", (user_id,))
            user = await cur.fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

async def get_current_user_or_api_key(request: Request) -> dict:
    """支持JWT token或API key认证"""
    auth = request.headers.get("Authorization", "")
    token = None
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
    else:
        token = request.query_params.get("token")
    
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token or API key")
    
    # 尝试作为JWT解析
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub", 0))
        
        async with request.app.state.db_pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT id, email, name, role FROM users WHERE id = %s", (user_id,))
                user = await cur.fetchone()
        
        if user:
            user_dict = dict(user)
            user_dict['auth_type'] = 'jwt'
            user_dict['scopes'] = ['full']  # JWT has full access
            return user_dict
    except:
        pass
    
    # 尝试作为API key解析
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT ak.user_id, ak.scopes, u.email, u.name, u.role
                FROM user_api_keys ak
                JOIN users u ON u.id = ak.user_id
                WHERE ak.key_hash = %s
                """,
                (key_hash,)
            )
            api_key_record = await cur.fetchone()
            
            if api_key_record:
                # 更新最后使用时间
                await cur.execute(
                    "UPDATE user_api_keys SET last_used = CURRENT_TIMESTAMP WHERE key_hash = %s",
                    (key_hash,)
                )
                
                scopes = api_key_record['scopes'].split(',') if api_key_record['scopes'] else []
                return {
                    'id': api_key_record['user_id'],
                    'email': api_key_record['email'],
                    'name': api_key_record['name'],
                    'role': api_key_record['role'],
                    'auth_type': 'api_key',
                    'scopes': scopes
                }
    
    raise HTTPException(status_code=401, detail="Invalid token or API key")

def require_scope(required_scope: str):
    """检查用户是否有特定权限"""
    async def checker(user: dict = Depends(get_current_user_or_api_key)) -> dict:
        if user.get('auth_type') == 'jwt':
            return user  # JWT has full access
        
        scopes = user.get('scopes', [])
        if required_scope in ('reading', 'cloud') and ('read' in scopes or 'write' in scopes):
            return user
        if required_scope not in scopes and 'full' not in scopes:
            raise HTTPException(status_code=403, detail=f"Scope '{required_scope}' required")
        return user
    return checker

async def require_admin(user: dict) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user

async def get_user_group_roles(pool: AsyncConnectionPool, user_id: int) -> dict:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT group_id, role FROM group_members WHERE user_id = %s", (user_id,))
            rows = await cur.fetchall()
    return {row["group_id"]: row["role"] for row in rows}

async def is_group_admin(pool: AsyncConnectionPool, user_id: int, group_id: int) -> bool:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM group_members WHERE user_id = %s AND group_id = %s AND role = 'admin'",
                (user_id, group_id)
            )
            return await cur.fetchone() is not None

def normalize_tags(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, list):
        parts = raw
    else:
        return []

    seen = set()
    tags: List[str] = []
    for item in parts:
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        tags.append(name)
    return tags

def normalize_data_types(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, list):
        parts = raw
    else:
        return []

    seen = set()
    types: List[str] = []
    for item in parts:
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        types.append(name[:32])
    return types

def generate_verification_code(length: int = 6) -> str:
    if length < 4:
        length = 4
    digits = "0123456789"
    return "".join(secrets.choice(digits) for _ in range(length))

async def set_item_tags(
    pool: AsyncConnectionPool,
    item_id: int,
    user_id: int,
    visibility: str,
    group_id: Optional[int],
    tags: List[str]
) -> None:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("DELETE FROM item_tags WHERE item_id = %s", (item_id,))
            if not tags:
                return

            for name in tags:
                if visibility == "group" and group_id:
                    await cur.execute(
                        "SELECT id FROM tags WHERE visibility = 'group' AND group_id = %s AND name = %s",
                        (group_id, name)
                    )
                    row = await cur.fetchone()
                    if not row:
                        await cur.execute(
                            "INSERT INTO tags (name, owner_user_id, visibility, group_id) VALUES (%s, %s, 'group', %s) RETURNING id",
                            (name, user_id, group_id)
                        )
                        row = await cur.fetchone()
                else:
                    await cur.execute(
                        "SELECT id FROM tags WHERE visibility = 'private' AND owner_user_id = %s AND name = %s",
                        (user_id, name)
                    )
                    row = await cur.fetchone()
                    if not row:
                        await cur.execute(
                            "INSERT INTO tags (name, owner_user_id, visibility) VALUES (%s, %s, 'private') RETURNING id",
                            (name, user_id)
                        )
                        row = await cur.fetchone()

                if row:
                    await cur.execute(
                        "INSERT INTO item_tags (item_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (item_id, row["id"])
                    )

def safe_filename(name: str) -> str:
    name = os.path.basename(name).strip()
    if not name:
        return "file"
    for ch in ["/", "\\", ".."]:
        name = name.replace(ch, "_")
    return name

def cloud_storage_path(user_id: int, original_name: str) -> str:
    ext = os.path.splitext(original_name)[1]
    storage_name = f"{uuid.uuid4().hex}{ext}"
    user_dir = os.path.join(CLOUD_STORAGE_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, storage_name)

def save_upload_stream(upload: UploadFile, dest_path: str) -> dict:
    hasher = hashlib.sha256()
    size = 0
    with open(dest_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    return {"size": size, "checksum": hasher.hexdigest()}

def ensure_within_storage(path: str) -> bool:
    real_path = os.path.realpath(path)
    return real_path.startswith(CLOUD_STORAGE_DIR)

async def ocr_with_ai(image_b64: str) -> str:
    base_url = os.getenv("AI_BASE_URL", "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail="AI_BASE_URL not configured")
    model = OCR_MODEL or os.getenv("VISION_MODEL", "").strip() or "gpt-4o-mini"
    endpoint = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized OCR engine. Extract text from images exactly as written. \n1. ONLY output the extracted text.\n2. Do NOT add any punctuation like '}' or words like 'Note' at the end.\n3. Do NOT attempt to complete, fix, or format the text beyond basic layout.\n4. Use Markdown tables if you see a table.\n5. If the image contains a snippet like 'in a far away land...', simply output that text. NEVER assume it's code or needs closure."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image exactly as it is. NO extra characters at the end."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, json=payload, headers={"Authorization": f"Bearer {os.getenv('AI_API_KEY')}"}, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            # If the model wraps it in markdown code blocks like ```markdown ... ```, strip them
            if content.startswith("```markdown") and content.endswith("```"):
                content = content[11:-3].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
            return content
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

async def get_cloud_item_for_user(request: Request, item_id: int, user: dict) -> dict:
    group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT * FROM cloud_items
                WHERE id = %s
                  AND (owner_user_id = %s OR (visibility = 'group' AND group_id = ANY(%s)))
                """,
                (item_id, user["id"], list(group_roles.keys()))
            )
            item = await cur.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail="Cloud item not found")
    return dict(item)

async def ensure_quick_send_folder(request: Request, owner_user_id: int) -> int:
    folder_name = "quick_send"
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id
                FROM cloud_items
                WHERE owner_user_id = %s
                  AND item_type = 'folder'
                  AND name = %s
                  AND parent_id IS NULL
                  AND visibility = 'private'
                ORDER BY id ASC
                LIMIT 1
                """,
                (owner_user_id, folder_name)
            )
            row = await cur.fetchone()
            if row:
                return int(row["id"])

            await cur.execute(
                """
                INSERT INTO cloud_items (owner_user_id, parent_id, name, item_type, visibility, group_id, tags)
                VALUES (%s, NULL, %s, 'folder', 'private', NULL, %s)
                RETURNING id
                """,
                (owner_user_id, folder_name, ["quick_send"])
            )
            row = await cur.fetchone()
            return int(row["id"])

# --- 4. 鉴权与用户组 ---

async def get_system_setting(request: Request, key: str, default=None):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT value FROM system_settings WHERE key = %s", (key,))
            row = await cur.fetchone()
            return row["value"] if row else default

def require_admin():
    async def dependency(payload: dict = Depends(get_current_user)):
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin privileges required")
        return payload
    return dependency

@app.post("/api/auth/register")
async def api_register(request: Request, data: dict):
    # Check registration toggle
    reg_setting = await get_system_setting(request, "registration", {"allow": True})
    if not reg_setting.get("allow", True):
        raise HTTPException(status_code=403, detail="Registration is disabled by administrator")

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()
    invite_code = (data.get("invite_code") or "").strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    pw = hash_password(password)

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if await cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already exists")

            await cur.execute(
                "INSERT INTO users (email, name, password_hash, password_salt) VALUES (%s, %s, %s, %s) RETURNING id, role",
                (email, name or None, pw["hash"], pw["salt"])
            )
            user = await cur.fetchone()

            if invite_code:
                now = datetime.now(timezone.utc)
                await cur.execute(
                    "SELECT code, group_id, expires_at, max_uses, uses FROM group_invites WHERE code = %s",
                    (invite_code,)
                )
                invite = await cur.fetchone()
                if not invite:
                    raise HTTPException(status_code=400, detail="Invalid invite code")
                if invite["expires_at"] and invite["expires_at"] < now:
                    raise HTTPException(status_code=400, detail="Invite code expired")
                if invite["uses"] >= invite["max_uses"]:
                    raise HTTPException(status_code=400, detail="Invite code exhausted")

                await cur.execute(
                    "INSERT INTO group_members (user_id, group_id, role) VALUES (%s, %s, 'member') ON CONFLICT DO NOTHING",
                    (user["id"], invite["group_id"])
                )
                await cur.execute(
                    "UPDATE group_invites SET uses = uses + 1 WHERE code = %s",
                    (invite_code,)
                )

    token = create_access_token(user["id"], user["role"])
    return {"token": token, "user": {"id": user["id"], "email": email, "name": name, "role": user["role"]}}

@app.post("/api/auth/login")
async def api_login(request: Request, data: dict):
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, email, name, role, password_hash, password_salt FROM users WHERE email = %s",
                (email,)
            )
            user = await cur.fetchone()

    if not user or not verify_password(password, user["password_salt"], user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user["id"], user["role"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"], "name": user["name"], "role": user["role"]}}

@app.get("/api/auth/me")
async def api_me(user: dict = Depends(get_current_user)):
    return {"user": user}

@app.get("/api/groups")
async def api_groups(request: Request, user: dict = Depends(get_current_user)):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT gm.group_id, gm.role, g.name
                FROM group_members gm
                JOIN groups g ON g.id = gm.group_id
                WHERE gm.user_id = %s
                ORDER BY g.name
            """, (user["id"],))
            rows = await cur.fetchall()

    return {"groups": [dict(row) for row in rows]}

@app.get("/api/tags")
async def api_tags(request: Request, user: dict = Depends(get_current_user)):
    group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
    group_ids = list(group_roles.keys())
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, name, visibility, group_id
                FROM tags
                WHERE (visibility = 'private' AND owner_user_id = %s)
                   OR (visibility = 'group' AND group_id = ANY(%s))
                ORDER BY name
                """,
                (user["id"], group_ids)
            )
            rows = await cur.fetchall()

    return {"tags": [dict(row) for row in rows]}

@app.post("/api/groups")
async def api_create_group(request: Request, data: dict, user: dict = Depends(get_current_user)):
    name = (data.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Group name is required")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO groups (name, owner_user_id) VALUES (%s, %s) RETURNING id",
                (name, user["id"])
            )
            group = await cur.fetchone()
            await cur.execute(
                "INSERT INTO group_members (user_id, group_id, role) VALUES (%s, %s, 'admin')",
                (user["id"], group["id"])
            )

    return {"status": "success", "group_id": group["id"]}

@app.post("/api/groups/join")
async def api_join_group(request: Request, data: dict, user: dict = Depends(get_current_user)):
    invite_code = (data.get("invite_code") or "").strip()
    if not invite_code:
        raise HTTPException(status_code=400, detail="Invite code is required")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            now = datetime.now(timezone.utc)
            await cur.execute(
                "SELECT code, group_id, expires_at, max_uses, uses FROM group_invites WHERE code = %s",
                (invite_code,)
            )
            invite = await cur.fetchone()
            if not invite:
                raise HTTPException(status_code=400, detail="Invalid invite code")
            if invite["expires_at"] and invite["expires_at"] < now:
                raise HTTPException(status_code=400, detail="Invite code expired")
            if invite["uses"] >= invite["max_uses"]:
                raise HTTPException(status_code=400, detail="Invite code exhausted")

            await cur.execute(
                "INSERT INTO group_members (user_id, group_id, role) VALUES (%s, %s, 'member') ON CONFLICT DO NOTHING",
                (user["id"], invite["group_id"])
            )
            await cur.execute(
                "UPDATE group_invites SET uses = uses + 1 WHERE code = %s",
                (invite_code,)
            )

    return {"status": "success"}

@app.post("/api/groups/{group_id}/invites")
async def api_create_invite(request: Request, group_id: int, data: dict, user: dict = Depends(get_current_user)):
    max_uses = int(data.get("max_uses") or 20)
    hours = int(data.get("expires_in_hours") or 168)
    now = datetime.now(timezone.utc)
    expires_at = now if hours <= 0 else now + timedelta(hours=hours)

    if not await is_group_admin(request.app.state.db_pool, user["id"], group_id) and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Group admin required")

    invite_code = secrets.token_urlsafe(12)
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO group_invites (code, group_id, created_by, expires_at, max_uses) VALUES (%s, %s, %s, %s, %s)",
                (invite_code, group_id, user["id"], expires_at, max_uses)
            )

    return {"status": "success", "invite_code": invite_code, "expires_at": expires_at}

@app.post("/api/groups/{group_id}/leave")
async def api_leave_group(request: Request, group_id: int, user: dict = Depends(get_current_user)):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT role FROM group_members WHERE user_id = %s AND group_id = %s",
                (user["id"], group_id)
            )
            member = await cur.fetchone()
            if not member:
                raise HTTPException(status_code=404, detail="Not a group member")

            if member["role"] == "admin":
                await cur.execute(
                    "SELECT COUNT(*) AS cnt FROM group_members WHERE group_id = %s AND role = 'admin'",
                    (group_id,)
                )
                admins = await cur.fetchone()
                if admins and admins["cnt"] <= 1:
                    raise HTTPException(status_code=400, detail="Last admin cannot leave")

            await cur.execute(
                "DELETE FROM group_members WHERE user_id = %s AND group_id = %s",
                (user["id"], group_id)
            )

    return {"status": "success"}

@app.get("/api/tags")
async def api_get_tags(request: Request, user: dict = Depends(get_current_user)):
    group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
    group_ids = list(group_roles.keys())
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, name, visibility, group_id 
                FROM tags 
                WHERE (owner_user_id = %s AND visibility = 'private')
                   OR (group_id = ANY(%s) AND visibility = 'group')
                ORDER BY name ASC
            """, (user["id"], group_ids))
            return {"tags": await cur.fetchall()}

@app.put("/api/tags/{tag_id}")
async def api_rename_tag(request: Request, tag_id: int, data: dict, user: dict = Depends(get_current_user)):
    new_name = (data.get("name") or "").strip()
    if not new_name: raise HTTPException(status_code=400, detail="Name required")
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM tags WHERE id = %s", (tag_id,))
            tag = await cur.fetchone()
            if not tag: raise HTTPException(status_code=404, detail="Tag not found")
            if tag["visibility"] == 'private':
                if tag["owner_user_id"] != user["id"]: raise HTTPException(status_code=403)
            else:
                if not await is_group_admin(request.app.state.db_pool, user["id"], tag["group_id"]):
                    raise HTTPException(status_code=403)
            await cur.execute("UPDATE tags SET name = %s WHERE id = %s", (new_name, tag_id))
            return {"status": "success"}

@app.delete("/api/tags/{tag_id}")
async def api_delete_tag(request: Request, tag_id: int, user: dict = Depends(get_current_user)):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM tags WHERE id = %s", (tag_id,))
            tag = await cur.fetchone()
            if not tag: raise HTTPException(status_code=404, detail="Tag not found")
            if tag["visibility"] == 'private':
                if tag["owner_user_id"] != user["id"]: raise HTTPException(status_code=403)
            else:
                if not await is_group_admin(request.app.state.db_pool, user["id"], tag["group_id"]):
                    raise HTTPException(status_code=403)
            await cur.execute("DELETE FROM tags WHERE id = %s", (tag_id,))
            return {"status": "success"}

@app.post("/api/groups/{group_id}/members/{member_id}/role")
async def api_update_member_role(
    request: Request,
    group_id: int,
    member_id: int,
    data: dict,
    user: dict = Depends(get_current_user)
):
    new_role = (data.get("role") or "").strip()
    if new_role not in ("member", "admin"):
        raise HTTPException(status_code=400, detail="Role must be member or admin")

    if not await is_group_admin(request.app.state.db_pool, user["id"], group_id) and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Group admin required")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE group_members SET role = %s WHERE user_id = %s AND group_id = %s",
                (new_role, member_id, group_id)
            )

    return {"status": "success"}

# --- 4. Web 路由接口 ---

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/app")
async def app_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

@app.get("/groups")
async def groups_page(request: Request):
    return templates.TemplateResponse("groups.html", {"request": request})

@app.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/detail")
async def detail_page(request: Request):
    return templates.TemplateResponse("detail.html", {"request": request})

@app.get("/reading")
async def reading_page(request: Request):
    return templates.TemplateResponse("reading.html", {"request": request})

@app.get("/cloud")
async def cloud_page(request: Request):
    return templates.TemplateResponse("cloud.html", {"request": request})

@app.get("/mini")
async def mini_page(request: Request):
    return templates.TemplateResponse("mini.html", {"request": request})

@app.get("/mini/quick-send")
async def mini_quick_send_page(request: Request):
    return templates.TemplateResponse("mini_quick_send.html", {"request": request})

@app.get("/quick")
async def quick_page(request: Request):
    return templates.TemplateResponse("quick.html", {"request": request})

@app.get("/api/v1/mini/settings")
async def api_v1_mini_get_settings(
    request: Request,
    user: dict = Depends(require_scope('read'))
):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT auto_provide FROM mini_user_settings WHERE user_id = %s",
                (user["id"],)
            )
            row = await cur.fetchone()
            if not row:
                await cur.execute(
                    "INSERT INTO mini_user_settings (user_id, auto_provide) VALUES (%s, FALSE)",
                    (user["id"],)
                )
                auto_provide = False
            else:
                auto_provide = bool(row["auto_provide"])
    return {"status": "success", "data": {"auto_provide": auto_provide}}

@app.put("/api/v1/mini/settings")
async def api_v1_mini_update_settings(
    request: Request,
    data: dict,
    user: dict = Depends(require_scope('read'))
):
    auto_provide = bool(data.get("auto_provide"))
    async with request.app.state.db_pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO mini_user_settings (user_id, auto_provide, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id)
            DO UPDATE SET auto_provide = EXCLUDED.auto_provide, updated_at = CURRENT_TIMESTAMP
            """,
            (user["id"], auto_provide)
        )
    return {"status": "success", "data": {"auto_provide": auto_provide}}

@app.post("/api/v1/mini/data-requests")
async def api_v1_mini_create_request(
    request: Request,
    data: dict,
    user: dict = Depends(require_scope('read'))
):
    app_name = (data.get("app_name") or "MiniApp").strip()[:60]
    data_types = normalize_data_types(data.get("data_types"))
    purpose = (data.get("purpose") or "").strip()[:240]

    if not data_types:
        raise HTTPException(status_code=400, detail="data_types is required")

    auto_provide = False
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT auto_provide FROM mini_user_settings WHERE user_id = %s",
                (user["id"],)
            )
            row = await cur.fetchone()
            if row:
                auto_provide = bool(row["auto_provide"])

    status = "approved" if auto_provide else "pending"
    grant_token = secrets.token_urlsafe(16) if auto_provide else None
    decided_at = datetime.now(timezone.utc) if auto_provide else None

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO mini_data_requests (owner_user_id, app_name, data_types, purpose, status, grant_token, decided_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (user["id"], app_name, data_types, purpose, status, grant_token, decided_at)
            )
            row = await cur.fetchone()

    return {
        "status": "success",
        "data": {
            "id": row["id"],
            "status": status,
            "grant_token": grant_token,
            "created_at": row["created_at"].isoformat() if row and row.get("created_at") else None
        }
    }

@app.get("/api/v1/mini/data-requests")
async def api_v1_mini_list_requests(
    request: Request,
    status: Optional[str] = Query(None),
    user: dict = Depends(require_scope('read'))
):
    query = "SELECT * FROM mini_data_requests WHERE owner_user_id = %s"
    params = [user["id"]]
    if status:
        query += " AND status = %s"
        params.append(status)
    query += " ORDER BY created_at DESC"

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, tuple(params))
            rows = await cur.fetchall()

    results = [dict(r) for r in rows]
    for r in results:
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
        if isinstance(r.get("decided_at"), datetime):
            r["decided_at"] = r["decided_at"].isoformat()
    return {"status": "success", "data": results}

@app.post("/api/v1/mini/data-requests/{request_id}/approve")
async def api_v1_mini_approve_request(
    request: Request,
    request_id: int,
    data: dict,
    user: dict = Depends(require_scope('read'))
):
    auto_provide = bool(data.get("auto_provide"))
    note = (data.get("note") or "").strip()[:240]
    grant_token = secrets.token_urlsafe(16)

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                UPDATE mini_data_requests
                SET status = 'approved', decision_note = %s, grant_token = %s, decided_at = CURRENT_TIMESTAMP
                WHERE id = %s AND owner_user_id = %s AND status = 'pending'
                RETURNING id
                """,
                (note, grant_token, request_id, user["id"])
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Request not found or already decided")

            if auto_provide:
                await cur.execute(
                    """
                    INSERT INTO mini_user_settings (user_id, auto_provide, updated_at)
                    VALUES (%s, TRUE, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id)
                    DO UPDATE SET auto_provide = TRUE, updated_at = CURRENT_TIMESTAMP
                    """,
                    (user["id"],)
                )

    return {"status": "success", "data": {"id": request_id, "grant_token": grant_token}}

@app.post("/api/v1/mini/data-requests/{request_id}/deny")
async def api_v1_mini_deny_request(
    request: Request,
    request_id: int,
    data: dict,
    user: dict = Depends(require_scope('read'))
):
    note = (data.get("note") or "").strip()[:240]
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                UPDATE mini_data_requests
                SET status = 'denied', decision_note = %s, decided_at = CURRENT_TIMESTAMP
                WHERE id = %s AND owner_user_id = %s AND status = 'pending'
                RETURNING id
                """,
                (note, request_id, user["id"])
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Request not found or already decided")
    return {"status": "success", "data": {"id": request_id}}

@app.get("/api/v1/mini/data")
async def api_v1_mini_get_data(
    request: Request,
    grant_token: str = Query(...),
    types: Optional[str] = Query(None)
):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT owner_user_id, data_types
                FROM mini_data_requests
                WHERE grant_token = %s AND status = 'approved'
                """,
                (grant_token,)
            )
            req = await cur.fetchone()
            if not req:
                raise HTTPException(status_code=403, detail="Invalid grant token")

            user_id = int(req["owner_user_id"])
            requested_types = normalize_data_types(types) or (req.get("data_types") or [])

            data: dict = {"types": requested_types}

            if "profile" in requested_types:
                await cur.execute("SELECT email, name FROM users WHERE id = %s", (user_id,))
                profile = await cur.fetchone()
                data["profile"] = {
                    "email": profile["email"],
                    "name": profile["name"]
                } if profile else None

            if "stats" in requested_types:
                await cur.execute("SELECT COUNT(*) AS cnt FROM intelligence_vault WHERE owner_user_id = %s", (user_id,))
                iv = await cur.fetchone()
                await cur.execute("SELECT COUNT(*) AS cnt FROM cloud_items WHERE owner_user_id = %s AND item_type = 'file'", (user_id,))
                files = await cur.fetchone()
                await cur.execute("SELECT COUNT(*) AS cnt FROM reading_list WHERE owner_user_id = %s", (user_id,))
                reading = await cur.fetchone()
                data["stats"] = {
                    "vault_items": int(iv["cnt"]),
                    "cloud_files": int(files["cnt"]),
                    "reading_items": int(reading["cnt"])
                }

    return {"status": "success", "data": data}

@app.post("/api/v1/mini/upload-sessions")
async def api_v1_mini_create_upload_session(
    request: Request,
    data: Optional[dict] = None,
    user: dict = Depends(require_scope('cloud'))
):
    ttl_minutes = 30
    if isinstance(data, dict) and data.get("ttl_minutes") is not None:
        try:
            ttl_minutes = int(data.get("ttl_minutes"))
        except (ValueError, TypeError):
            ttl_minutes = 30
    ttl_minutes = max(5, min(ttl_minutes, 180))
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)

    code = None
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            for _ in range(6):
                candidate = generate_verification_code(6)
                await cur.execute(
                    "SELECT 1 FROM mini_upload_sessions WHERE code = %s",
                    (candidate,)
                )
                if not await cur.fetchone():
                    code = candidate
                    break
            if not code:
                raise HTTPException(status_code=500, detail="Unable to allocate code")

            await cur.execute(
                """
                INSERT INTO mini_upload_sessions (code, owner_user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (code, user["id"], expires_at)
            )

    return {
        "status": "success",
        "data": {
            "code": code,
            "expires_at": expires_at.isoformat(),
            "upload_url": f"/mini/quick-send?code={code}"
        }
    }

@app.get("/api/v1/mini/upload-sessions/{code}")
async def api_v1_mini_get_upload_session(
    request: Request,
    code: str
):
    if not code.isdigit() or len(code) != 6:
        raise HTTPException(status_code=400, detail="Invalid code")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT code, expires_at, status FROM mini_upload_sessions WHERE code = %s",
                (code,)
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")

    expires_at = row.get("expires_at")
    expired = False
    if isinstance(expires_at, datetime):
        expired = expires_at < datetime.now(timezone.utc)

    return {
        "status": "success",
        "data": {
            "code": row["code"],
            "status": row["status"],
            "expired": expired,
            "expires_at": expires_at.isoformat() if isinstance(expires_at, datetime) else None
        }
    }

@app.get("/api/v1/mini/upload-sessions/{code}/files")
async def api_v1_mini_list_upload_files(
    request: Request,
    code: str,
    user: dict = Depends(require_scope('cloud'))
):
    if not code.isdigit() or len(code) != 6:
        raise HTTPException(status_code=400, detail="Invalid code")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT owner_user_id FROM mini_upload_sessions WHERE code = %s",
                (code,)
            )
            session = await cur.fetchone()
            if not session or int(session["owner_user_id"]) != int(user["id"]):
                raise HTTPException(status_code=403, detail="Not allowed")

            await cur.execute(
                """
                SELECT id, cloud_item_id, filename, size, created_at
                FROM mini_upload_files
                WHERE session_code = %s
                ORDER BY created_at DESC
                """,
                (code,)
            )
            rows = await cur.fetchall()

    results = [dict(r) for r in rows]
    for r in results:
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
    return {"status": "success", "data": results}

@app.post("/api/v1/mini/upload")
async def api_v1_mini_upload_file(
    request: Request,
    code: str = Query(...),
    file: UploadFile = File(...)
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File is required")
    if not code.isdigit() or len(code) != 6:
        raise HTTPException(status_code=400, detail="Invalid code")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT owner_user_id, expires_at, status FROM mini_upload_sessions WHERE code = %s",
                (code,)
            )
            session = await cur.fetchone()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            expires_at = session.get("expires_at")
            if session.get("status") != "active":
                raise HTTPException(status_code=400, detail="Session inactive")
            if isinstance(expires_at, datetime) and expires_at < datetime.now(timezone.utc):
                raise HTTPException(status_code=400, detail="Session expired")

            owner_user_id = int(session["owner_user_id"])

            parent_id = await ensure_quick_send_folder(request, owner_user_id)

            filename = safe_filename(file.filename)
            mime_type = file.content_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            storage_path = cloud_storage_path(owner_user_id, filename)
            meta = save_upload_stream(file, storage_path)
            if not ensure_within_storage(storage_path):
                raise HTTPException(status_code=400, detail="Invalid storage path")

            await cur.execute(
                """
                INSERT INTO cloud_items (owner_user_id, parent_id, name, item_type, size, mime_type, storage_path,
                                         checksum, visibility, group_id, tags)
                VALUES (%s, %s, %s, 'file', %s, %s, %s, %s, 'private', NULL, %s)
                RETURNING id
                """,
                (owner_user_id, parent_id, filename, meta["size"], mime_type, storage_path, meta["checksum"], ["mini-ship", "quick_send"])
            )
            row = await cur.fetchone()
            item_id = row["id"]

            await cur.execute(
                """
                INSERT INTO cloud_versions (item_id, version_num, size, checksum, storage_path, created_by)
                VALUES (%s, 1, %s, %s, %s, %s)
                """,
                (item_id, meta["size"], meta["checksum"], storage_path, owner_user_id)
            )

            await cur.execute(
                """
                INSERT INTO mini_upload_files (session_code, cloud_item_id, filename, size)
                VALUES (%s, %s, %s, %s)
                """,
                (code, item_id, filename, meta["size"])
            )

            await cur.execute(
                "UPDATE mini_upload_sessions SET last_used = CURRENT_TIMESTAMP WHERE code = %s",
                (code,)
            )

    return {"status": "success", "data": {"id": item_id, "name": filename, "size": meta["size"]}}

# 异步背景任务：向量化单个项目
async def background_vectorize_item(pool: AsyncConnectionPool, item_id: int, text_to_vectorize: str):
    """后台任务：为单个情报项生成向量并更新数据库"""
    try:
        vec = await get_embedding(text_to_vectorize)
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE intelligence_vault SET embedding = %s::vector, vectorization_status = 'success' WHERE id = %s",
                (json.dumps(vec), item_id)
            )
        logger.info(f"✓ 项目 #{item_id} 向量化完成")
    except Exception as e:
        logger.error(f"✗ 项目 #{item_id} 向量化失败: {e}")
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE intelligence_vault SET vectorization_status = 'failed' WHERE id = %s",
                (item_id,)
            )

# 异步背景任务：全量重新向量化
async def background_revectorize(pool: AsyncConnectionPool):
    logger.info("开始执行异步全量重新向量化任务...")
    start_time = time.time()
    try:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT id, title, content FROM intelligence_vault")
                all_items = await cur.fetchall()
                
                count = 0
                for item in all_items:
                    # 构建向量化文本（标题 + 内容）
                    text = (item.get('title') or '') + ' ' + (item.get('content') or '')
                    text = text.strip()
                    
                    # 强制刷新向量 (这里不使用缓存逻辑，因为是"重构")
                    base_url = os.getenv("AI_BASE_URL").rstrip('/')
                    endpoint = f"{base_url}/embeddings"
                    headers = {"Authorization": f"Bearer {os.getenv('AI_API_KEY')}"}
                    payload = {"model": os.getenv("EMBEDDING_MODEL"), "input": text}
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(endpoint, json=payload, headers=headers, timeout=10.0)
                        if response.status_code == 200:
                            vec = response.json()['data'][0]['embedding']
                            await conn.execute(
                                "UPDATE intelligence_vault SET embedding = %s::vector, vectorization_status = 'success' WHERE id = %s",
                                (json.dumps(vec), item['id'])
                            )
                        else:
                            await conn.execute(
                                "UPDATE intelligence_vault SET vectorization_status = 'failed' WHERE id = %s",
                                (item['id'],)
                            )
                    
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"后台进度: {count}/{len(all_items)}")
            
        duration = (time.time() - start_time) * 1000
        logger.info(f">>> 异步全量重新向量化完成 | 总数: {count} | 总耗时: {duration:.2f}ms")
    except Exception as e:
        logger.error(f"后台批量更新失败: {e}")

@app.post("/api/v1/admin/revectorize-all")
async def api_v1_revectorize_all(
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_scope('admin'))
):
    """管理员工具：全量重新向量化"""
    # 立即返回，后台执行
    background_tasks.add_task(background_revectorize, request.app.state.db_pool)
    return {"status": "success", "message": "Revectorization task started in background"}

# --- REST API v1 (新版API with proper REST conventions) ---

@app.get("/api/v1/items")
async def api_v1_list_items(
    request: Request,
    type: str = 'all',
    q: str = '',
    page: int = 1,
    per_page: int = Query(default=10, le=100),
    show_archived: bool = False,
    favorited_only: bool = False,
    user: dict = Depends(require_scope('read'))
):
    """列出情报项目（支持搜索、归档过滤和分页）"""
    with Profiler() as p_total:
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        
        offset = (page - 1) * per_page
        results = []
        
        query_vec = None
        if type == 'ai' and q:
            query_vec = await get_embedding(q)
        
        async with request.app.state.db_pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                access_sql = "(iv.owner_user_id = %s OR (iv.visibility = 'group' AND iv.group_id = ANY(%s)))"
                archive_sql = "" if show_archived else " AND iv.is_archived = FALSE"
                favorite_sql = " AND iv.is_favorited = TRUE" if favorited_only else ""
                
                if type == 'ai' and query_vec:
                    await cur.execute("""
                        SELECT iv.id, iv.title, 
                               LEFT(iv.content, 200) as content_preview,
                               iv.created_at, iv.visibility, iv.group_id,
                               iv.vectorization_status, iv.is_archived, iv.is_favorited,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags,
                               1 - (iv.embedding <=> %s::vector) AS score
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE """ + access_sql + archive_sql + favorite_sql + """
                        ORDER BY score DESC LIMIT %s OFFSET %s
                    """, (json.dumps(query_vec), user["id"], list(group_roles.keys()), per_page, offset))
                elif type == 'key' and q:
                    await cur.execute("""
                        SELECT iv.id, iv.title,
                               LEFT(iv.content, 200) as content_preview,
                               iv.created_at, iv.visibility, iv.group_id,
                               iv.vectorization_status, iv.is_archived, iv.is_favorited,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE (iv.content ILIKE %s OR iv.title ILIKE %s) AND """ + access_sql + archive_sql + favorite_sql + """
                        ORDER BY iv.is_favorited DESC, iv.created_at DESC LIMIT %s OFFSET %s
                    """, (f'%{q}%', f'%{q}%', user["id"], list(group_roles.keys()), per_page, offset))
                else:
                    await cur.execute("""
                        SELECT iv.id, iv.title,
                               LEFT(iv.content, 200) as content_preview,
                               iv.created_at, iv.visibility, iv.group_id,
                               iv.vectorization_status, iv.is_archived, iv.is_favorited,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE """ + access_sql + archive_sql + favorite_sql + """
                        ORDER BY iv.is_favorited DESC, iv.created_at DESC LIMIT %s OFFSET %s
                    """, (user["id"], list(group_roles.keys()), per_page, offset))
                
                rows = await cur.fetchall()
                results = [dict(row) for row in rows]
        
        for r in results:
            if isinstance(r['created_at'], datetime):
                r['created_at'] = r['created_at'].isoformat()
    
    logger.info(f">>> v1 搜索请求总响应时间: {p_total.duration:.2f}ms")
    return {
        "status": "success",
        "data": results,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "has_more": len(results) == per_page
        }
    }

@app.get("/api/v1/items/{item_id}")
async def api_v1_get_item(
    request: Request,
    item_id: int,
    user: dict = Depends(require_scope('read'))
):
    """获取单个情报项目详情"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT iv.id, iv.title, iv.content, iv.created_at, iv.visibility, iv.group_id,
                       iv.vectorization_status, iv.is_archived, iv.is_favorited,
                       iv.owner_user_id, u.name as owner_name,
                       g.name AS group_name,
                       ARRAY(
                           SELECT t.name FROM item_tags it
                           JOIN tags t ON t.id = it.tag_id
                           WHERE it.item_id = iv.id
                           ORDER BY t.name
                       ) AS tags
                FROM intelligence_vault iv
                LEFT JOIN groups g ON g.id = iv.group_id
                LEFT JOIN users u ON u.id = iv.owner_user_id
                WHERE iv.id = %s
            """, (item_id,))
            item = await cur.fetchone()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item_dict = dict(item)
    
    # 权限检查
    group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
    if item_dict['owner_user_id'] != user["id"]:
        if item_dict['visibility'] != 'group' or item_dict['group_id'] not in group_roles:
            raise HTTPException(status_code=403, detail="No access to this item")
    
    if isinstance(item_dict['created_at'], datetime):
        item_dict['created_at'] = item_dict['created_at'].isoformat()
    
    return {"status": "success", "data": item_dict}

@app.post("/api/v1/items")
async def api_v1_create_item(
    request: Request,
    background_tasks: BackgroundTasks,
    data: dict,
    user: dict = Depends(require_scope('write'))
):
    """创建新的情报项目（异步向量化）"""
    title = (data.get('title') or '').strip()
    content = data.get('content', '').strip()
    visibility = (data.get("visibility") or "private").strip()
    group_id = data.get("group_id")
    tags_input = data.get("tags")
    
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")
    if visibility not in ("private", "group"):
        raise HTTPException(status_code=400, detail="Visibility must be 'private' or 'group'")
    
    if visibility == "group":
        if not group_id:
            raise HTTPException(status_code=400, detail="group_id is required for group visibility")
        group_id = int(group_id)
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        if group_id not in group_roles:
            raise HTTPException(status_code=403, detail="Not a member of the group")
    else:
        group_id = None
    
    tags = normalize_tags(tags_input)
    
    # 先插入数据（embedding 为 NULL, status 为 pending），立即返回
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO intelligence_vault (title, content, embedding, owner_user_id, visibility, group_id, vectorization_status)
                VALUES (%s, %s, NULL, %s, %s, %s, 'pending')
                RETURNING id
                """,
                (title or None, content, user["id"], visibility, group_id)
            )
            row = await cur.fetchone()
            item_id = row["id"]
    
    await set_item_tags(request.app.state.db_pool, item_id, user["id"], visibility, group_id, tags)
    
    # 启动后台向量化任务
    text_to_vectorize = (title + ' ' + content) if title else content
    background_tasks.add_task(background_vectorize_item, request.app.state.db_pool, item_id, text_to_vectorize)
    
    logger.info(f"✓ 情报 #{item_id} 已创建，向量化任务已加入队列")
    return {"status": "success", "data": {"id": item_id}}

@app.post("/api/v1/items/from-link")
async def api_v1_create_item_from_link(
    request: Request,
    background_tasks: BackgroundTasks,
    data: dict,
    user: dict = Depends(require_scope('write'))
):
    """创建仅链接的情报项目（标题 + URL）"""
    title = (data.get('title') or '').strip()
    url = (data.get('url') or '').strip()
    note = (data.get('note') or '').strip()
    if not title or not url:
        raise HTTPException(status_code=400, detail="Title and URL are required")
    content = f"# {title}\n\n**来源**: {url}\n"
    if note:
        content += f"\n---\n\n{note}\n"

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO intelligence_vault (title, content, embedding, owner_user_id, visibility, vectorization_status)
                VALUES (%s, %s, NULL, %s, 'private', 'pending')
                RETURNING id
                """,
                (title, content, user["id"])
            )
            row = await cur.fetchone()
            item_id = row["id"]

    background_tasks.add_task(background_vectorize_item, request.app.state.db_pool, item_id, f"{title} {content}")
    return {"status": "success", "data": {"id": item_id}}

@app.put("/api/v1/items/{item_id}")
async def api_v1_update_item(
    request: Request,
    background_tasks: BackgroundTasks,
    item_id: int,
    data: dict,
    user: dict = Depends(require_scope('write'))
):
    """更新情报项目（异步向量化）"""
    title = (data.get('title') or '').strip()
    content = data.get('content', '').strip()
    visibility = (data.get("visibility") or "private").strip()
    group_id = data.get("group_id")
    tags_input = data.get("tags", None)
    
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")
    if visibility not in ("private", "group"):
        raise HTTPException(status_code=400, detail="Visibility must be 'private' or 'group'")
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, owner_user_id FROM intelligence_vault WHERE id = %s",
                (item_id,)
            )
            item = await cur.fetchone()
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            if item["owner_user_id"] != user["id"]:
                raise HTTPException(status_code=403, detail="No permission to update")
            
            if visibility == "group":
                if not group_id:
                    raise HTTPException(status_code=400, detail="group_id is required for group visibility")
                group_id = int(group_id)
                group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
                if group_id not in group_roles:
                    raise HTTPException(status_code=403, detail="Not a member of the group")
            else:
                group_id = None
            
            # 先更新数据（重置向量化状态），立即返回
            await cur.execute(
                """
                UPDATE intelligence_vault
                SET title = %s, content = %s, visibility = %s, group_id = %s, 
                    embedding = NULL, vectorization_status = 'pending'
                WHERE id = %s
                """,
                (title or None, content, visibility, group_id, item_id)
            )
    
    if tags_input is not None:
        tags = normalize_tags(tags_input)
        await set_item_tags(request.app.state.db_pool, item_id, user["id"], visibility, group_id, tags)
    
    # 启动后台向量化任务
    text_to_vectorize = (title + ' ' + content) if title else content
    background_tasks.add_task(background_vectorize_item, request.app.state.db_pool, item_id, text_to_vectorize)
    
    logger.info(f"✓ 情报 #{item_id} 已更新，向量化任务已加入队列")
    return {"status": "success"}

@app.delete("/api/v1/items/{item_id}")
async def api_v1_delete_item(
    request: Request,
    item_id: int,
    user: dict = Depends(require_scope('write'))
):
    """删除情报项目"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, owner_user_id FROM intelligence_vault WHERE id = %s",
                (item_id,)
            )
            item = await cur.fetchone()
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            if item["owner_user_id"] != user["id"]:
                raise HTTPException(status_code=403, detail="No permission to delete")
            
            await cur.execute("DELETE FROM intelligence_vault WHERE id = %s", (item_id,))
    
    return {"status": "success"}

@app.patch("/api/v1/items/{item_id}/status")
async def api_v1_update_item_status(
    request: Request,
    item_id: int,
    data: dict,
    user: dict = Depends(require_scope('write'))
):
    """快速更新情报项目状态（归档、收藏等），不触发向量化"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, owner_user_id FROM intelligence_vault WHERE id = %s",
                (item_id,)
            )
            item = await cur.fetchone()
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            if item["owner_user_id"] != user["id"]:
                raise HTTPException(status_code=403, detail="No permission to update")
            
            updates = []
            params = []
            
            if 'is_archived' in data:
                updates.append("is_archived = %s")
                params.append(bool(data['is_archived']))
            
            if 'is_favorited' in data:
                updates.append("is_favorited = %s")
                params.append(bool(data['is_favorited']))
            
            if updates:
                params.append(item_id)
                await cur.execute(
                    f"UPDATE intelligence_vault SET {', '.join(updates)} WHERE id = %s",
                    params
                )
    
    return {"status": "success"}

# --- Reading List API (稍后阅读) ---

@app.post("/api/v1/reading-list")
async def api_v1_create_reading_item(
    request: Request,
    data: dict,
    user: dict = Depends(require_scope('reading'))
):
    """添加稍后阅读项目"""
    title = (data.get('title') or '').strip()
    url = (data.get('url') or '').strip()
    content = (data.get('content') or '').strip()
    source = (data.get('source') or '').strip()
    cover_image = (data.get('cover_image') or '').strip()
    tags = data.get('tags', [])
    
    if not title or not url:
        raise HTTPException(status_code=400, detail="Title and URL are required")
    
    # 标签处理
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',') if t.strip()]
    elif not isinstance(tags, list):
        tags = []
    
    has_content = bool(content)
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO reading_list (title, url, content, source, cover_image, has_content, tags, owner_user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (title, url, content or None, source or None, cover_image or None, has_content, tags, user["id"])
            )
            row = await cur.fetchone()
            item_id = row["id"]
    
    return {"status": "success", "data": {"id": item_id}}

@app.get("/api/v1/reading-list")
async def api_v1_list_reading_items(
    request: Request,
    filter: str = 'all',  # all, unread, read, archived
    page: int = 1,
    per_page: int = Query(default=20, le=100),
    user: dict = Depends(require_scope('reading'))
):
    """获取阅读列表"""
    offset = (page - 1) * per_page
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            base_query = "FROM reading_list WHERE owner_user_id = %s"
            params = [user["id"]]
            
            if filter == 'unread':
                base_query += " AND is_read = FALSE AND is_archived = FALSE"
            elif filter == 'read':
                base_query += " AND is_read = TRUE AND is_archived = FALSE"
            elif filter == 'archived':
                base_query += " AND is_archived = TRUE"
            else:  # all
                base_query += " AND is_archived = FALSE"
            
            await cur.execute(
                f"""
                SELECT id, title, url, source, cover_image, has_content, is_read, is_archived, is_favorited, tags, created_at
                {base_query}
                ORDER BY is_favorited DESC, created_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [per_page, offset]
            )
            rows = await cur.fetchall()
            results = [dict(row) for row in rows]
    
    for r in results:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].isoformat()
    
    return {
        "status": "success",
        "data": results,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "has_more": len(results) == per_page
        }
    }

@app.get("/api/v1/reading-list/{item_id}")
async def api_v1_get_reading_item(
    request: Request,
    item_id: int,
    user: dict = Depends(require_scope('reading'))
):
    """获取单个阅读项详情"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT * FROM reading_list WHERE id = %s AND owner_user_id = %s
                """,
                (item_id, user["id"])
            )
            item = await cur.fetchone()
    
    if not item:
        raise HTTPException(status_code=404, detail="Reading item not found")
    
    item_dict = dict(item)
    if isinstance(item_dict.get('created_at'), datetime):
        item_dict['created_at'] = item_dict['created_at'].isoformat()
    if isinstance(item_dict.get('updated_at'), datetime):
        item_dict['updated_at'] = item_dict['updated_at'].isoformat()
    
    return {"status": "success", "data": item_dict}

@app.put("/api/v1/reading-list/{item_id}")
async def api_v1_update_reading_item(
    request: Request,
    item_id: int,
    data: dict,
    user: dict = Depends(require_scope('reading'))
):
    """更新阅读项（标记已读、归档等）"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id FROM reading_list WHERE id = %s AND owner_user_id = %s",
                (item_id, user["id"])
            )
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="Reading item not found")
            
            updates = []
            params = []
            
            if 'is_read' in data:
                updates.append("is_read = %s")
                params.append(bool(data['is_read']))
            
            if 'is_archived' in data:
                updates.append("is_archived = %s")
                params.append(bool(data['is_archived']))
            
            if 'is_favorited' in data:
                updates.append("is_favorited = %s")
                params.append(bool(data['is_favorited']))
            
            if 'tags' in data:
                tags = data['tags']
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',') if t.strip()]
                elif not isinstance(tags, list):
                    tags = []
                updates.append("tags = %s")
                params.append(tags)
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(item_id)
                await cur.execute(
                    f"UPDATE reading_list SET {', '.join(updates)} WHERE id = %s",
                    params
                )
    
    return {"status": "success"}

@app.delete("/api/v1/reading-list/{item_id}")
async def api_v1_delete_reading_item(
    request: Request,
    item_id: int,
    user: dict = Depends(require_scope('reading'))
):
    """删除阅读项"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id FROM reading_list WHERE id = %s AND owner_user_id = %s",
                (item_id, user["id"])
            )
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="Reading item not found")
            
            await cur.execute("DELETE FROM reading_list WHERE id = %s", (item_id,))
    
    return {"status": "success"}

@app.post("/api/v1/reading-list/{item_id}/save-to-vault")
async def api_v1_save_reading_to_vault(
    request: Request,
    background_tasks: BackgroundTasks,
    item_id: int,
    user: dict = Depends(require_scope('write'))
):
    """将阅读项保存到情报库"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # 获取阅读项
            await cur.execute(
                "SELECT * FROM reading_list WHERE id = %s AND owner_user_id = %s",
                (item_id, user["id"])
            )
            reading_item = await cur.fetchone()
            
            if not reading_item:
                raise HTTPException(status_code=404, detail="Reading item not found")
            
            # 创建情报项
            title = f"{reading_item['title']} (来源: {reading_item['source'] or reading_item['url']})"
            if reading_item['has_content']:
                content = f"# {reading_item['title']}\n\n**来源**: {reading_item['url']}\n\n---\n\n{reading_item['content']}"
            else:
                content = f"# {reading_item['title']}\n\n**来源**: {reading_item['url']}\n\n> 仅保存链接，无正文内容。"
            
            await cur.execute(
                """
                INSERT INTO intelligence_vault (title, content, embedding, owner_user_id, visibility, vectorization_status)
                VALUES (%s, %s, NULL, %s, 'private', 'pending')
                RETURNING id
                """,
                (title, content, user["id"])
            )
            vault_item = await cur.fetchone()
            vault_id = vault_item["id"]
            
            # 标记阅读项为已归档
            await cur.execute(
                "UPDATE reading_list SET is_archived = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (item_id,)
            )
    
    # 启动后台向量化任务
    text_to_vectorize = f"{title} {content}"
    background_tasks.add_task(background_vectorize_item, request.app.state.db_pool, vault_id, text_to_vectorize)
    
    return {"status": "success", "data": {"vault_id": vault_id}}

# --- Cloud Drive API (云盘) ---

@app.get("/api/v1/cloud/items")
async def api_v1_cloud_list_items(
    request: Request,
    parent_id: Optional[int] = None,
    filter_by: str = 'all',  # all, favorites, archived, shared
    q: str = '',
    page: int = 1,
    per_page: int = Query(default=50, le=200),
    user: dict = Depends(require_scope('cloud'))
):
    group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
    offset = (page - 1) * per_page
    params = [user["id"], list(group_roles.keys())]
    where_sql = "WHERE (owner_user_id = %s OR (visibility = 'group' AND group_id = ANY(%s)))"

    if parent_id is None:
        where_sql += " AND parent_id IS NULL"
    else:
        where_sql += " AND parent_id = %s"
        params.append(parent_id)

    if filter_by == 'favorites':
        where_sql += " AND is_favorited = TRUE AND is_archived = FALSE"
    elif filter_by == 'archived':
        where_sql += " AND is_archived = TRUE"
    elif filter_by == 'shared':
        where_sql += " AND is_shared = TRUE"
    else:
        where_sql += " AND is_archived = FALSE"

    if q:
        where_sql += " AND name ILIKE %s"
        params.append(f"%{q}%")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                SELECT id, name, item_type, size, mime_type, visibility, group_id,
                       is_shared, is_archived, is_favorited, tags, created_at, updated_at
                FROM cloud_items
                {where_sql}
                ORDER BY CASE WHEN item_type = 'folder' THEN 0 ELSE 1 END, name ASC
                LIMIT %s OFFSET %s
                """,
                params + [per_page, offset]
            )
            rows = await cur.fetchall()

    results = [dict(row) for row in rows]
    for r in results:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].isoformat()
        if isinstance(r.get('updated_at'), datetime):
            r['updated_at'] = r['updated_at'].isoformat()

    return {
        "status": "success",
        "data": results,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "has_more": len(results) == per_page
        }
    }

@app.post("/api/v1/cloud/folders")
async def api_v1_cloud_create_folder(
    request: Request,
    data: dict,
    user: dict = Depends(require_scope('cloud'))
):
    name = safe_filename((data.get('name') or '').strip())
    parent_id = data.get('parent_id')
    visibility = (data.get('visibility') or 'private').strip()
    group_id = data.get('group_id')
    tags = normalize_tags(data.get('tags'))

    if not name:
        raise HTTPException(status_code=400, detail="Folder name is required")
    if visibility not in ("private", "group"):
        raise HTTPException(status_code=400, detail="Visibility must be 'private' or 'group'")

    if visibility == "group":
        if not group_id:
            raise HTTPException(status_code=400, detail="group_id is required for group visibility")
        group_id = int(group_id)
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        if group_id not in group_roles:
            raise HTTPException(status_code=403, detail="Not a member of the group")
    else:
        group_id = None

    if parent_id is not None:
        parent = await get_cloud_item_for_user(request, int(parent_id), user)
        if parent.get('item_type') != 'folder':
            raise HTTPException(status_code=400, detail="Parent must be a folder")

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO cloud_items (owner_user_id, parent_id, name, item_type, visibility, group_id, tags)
                VALUES (%s, %s, %s, 'folder', %s, %s, %s)
                RETURNING id
                """,
                (user["id"], parent_id, name, visibility, group_id, tags)
            )
            row = await cur.fetchone()
    return {"status": "success", "data": {"id": row["id"]}}


@app.post("/api/v1/cloud/upload/chunk")
async def api_v1_upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(require_scope('cloud'))
):
    """上传文件分片"""
    # 安全检查 upload_id
    if not upload_id.isalnum():
         raise HTTPException(status_code=400, detail="Invalid upload_id")
         
    temp_dir = os.path.join(CLOUD_STORAGE_DIR, "temp", upload_id)
    os.makedirs(temp_dir, exist_ok=True)
    chunk_path = os.path.join(temp_dir, f"{chunk_index}")
    
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {"status": "success"}

@app.post("/api/v1/cloud/upload/merge")
async def api_v1_upload_merge(
    request: Request,
    upload_id: str = Form(...),
    filename: str = Form(...),
    total_chunks: int = Form(...),
    parent_id: Optional[int] = Form(None),
    visibility: str = Form("private"),
    group_id: Optional[int] = Form(None),
    item_id: Optional[int] = Form(None),
    tags: str = Form(""),
    user: dict = Depends(require_scope('cloud'))
):
    """合并分片并保存文件"""
    if not upload_id.isalnum():
         raise HTTPException(status_code=400, detail="Invalid upload_id")

    temp_dir = os.path.join(CLOUD_STORAGE_DIR, "temp", upload_id)
    if not os.path.exists(temp_dir):
        raise HTTPException(status_code=400, detail="Upload session not found")

    # 1. Merge files
    filename = safe_filename(filename)
    final_storage_path = cloud_storage_path(user["id"], filename)
    
    # Ensure dir exists (cloud_storage_path creates user dir but safeguard)
    os.makedirs(os.path.dirname(final_storage_path), exist_ok=True)

    try:
        with open(final_storage_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk_path = os.path.join(temp_dir, f"{i}")
                if not os.path.exists(chunk_path):
                    raise HTTPException(status_code=400, detail=f"Missing chunk {i}")
                with open(chunk_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise HTTPException(status_code=500, detail="Merge failed")
    finally:
        # Cleanup chunks
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 2. Calculate metadata
    mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    hasher = hashlib.sha256()
    size = 0
    with open(final_storage_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
            size += len(chunk)
    checksum = hasher.hexdigest()

    # 3. DB Insertion (Similar to api_v1_cloud_upload)
    
    # --- Logic for New Version ---
    if item_id:
        existing = await get_cloud_item_for_user(request, int(item_id), user)
        if existing.get('item_type') != 'file':
            raise HTTPException(status_code=400, detail="Only files can upload new versions")

        async with request.app.state.db_pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT COALESCE(MAX(version_num), 0) AS max_ver FROM cloud_versions WHERE item_id = %s",
                    (item_id,)
                )
                row = await cur.fetchone()
                next_ver = int(row["max_ver"]) + 1
                await cur.execute(
                    """
                    INSERT INTO cloud_versions (item_id, version_num, size, checksum, storage_path, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (item_id, next_ver, size, checksum, final_storage_path, user["id"])
                )
                await cur.execute(
                    """
                    UPDATE cloud_items
                    SET size = %s, checksum = %s, storage_path = %s, mime_type = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (size, checksum, final_storage_path, mime_type, item_id)
                )
        return {"status": "success", "data": {"id": int(item_id), "version": next_ver}}

    # --- Logic for New File ---
    if visibility not in ("private", "group"):
        raise HTTPException(status_code=400, detail="Visibility")
    if visibility == "group":
        if not group_id:
             raise HTTPException(status_code=400, detail="Group ID required")
        # Skipping strictly redundant group check for brevity (assumed handled by FE/Token role, but technically should check)
        # Re-adding check for safety:
        group_id = int(group_id)
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        if group_id not in group_roles:
            raise HTTPException(status_code=403, detail="Not group member")
    else:
        group_id = None

    if parent_id is not None:
        parent = await get_cloud_item_for_user(request, int(parent_id), user)
        if parent.get('item_type') != 'folder':
             raise HTTPException(status_code=400, detail="Parent not folder")

    tags_list = normalize_tags(tags)

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO cloud_items (owner_user_id, parent_id, name, item_type, size, mime_type, storage_path,
                                         checksum, visibility, group_id, tags)
                VALUES (%s, %s, %s, 'file', %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user["id"], parent_id, filename, size, mime_type, final_storage_path, checksum, visibility, group_id, tags_list)
            )
            row = await cur.fetchone()
            new_id = row["id"]

    return {"status": "success", "data": {"id": new_id, "name": filename}}

@app.post("/api/v1/cloud/upload")
async def api_v1_cloud_upload(
    request: Request,
    file: UploadFile = File(...),
    parent_id: Optional[int] = Form(None),
    visibility: str = Form('private'),
    group_id: Optional[int] = Form(None),
    item_id: Optional[int] = Form(None),
    tags: Optional[str] = Form(None),
    user: dict = Depends(require_scope('cloud'))
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    filename = safe_filename(file.filename)
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream'

    if item_id:
        existing = await get_cloud_item_for_user(request, int(item_id), user)
        if existing.get('item_type') != 'file':
            raise HTTPException(status_code=400, detail="Only files can upload new versions")
        storage_path = cloud_storage_path(user["id"], filename)
        meta = save_upload_stream(file, storage_path)
        if not ensure_within_storage(storage_path):
            raise HTTPException(status_code=400, detail="Invalid storage path")

        async with request.app.state.db_pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT COALESCE(MAX(version_num), 0) AS max_ver FROM cloud_versions WHERE item_id = %s",
                    (item_id,)
                )
                row = await cur.fetchone()
                next_ver = int(row["max_ver"]) + 1
                await cur.execute(
                    """
                    INSERT INTO cloud_versions (item_id, version_num, size, checksum, storage_path, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (item_id, next_ver, meta["size"], meta["checksum"], storage_path, user["id"])
                )
                await cur.execute(
                    """
                    UPDATE cloud_items
                    SET size = %s, checksum = %s, storage_path = %s, mime_type = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (meta["size"], meta["checksum"], storage_path, mime_type, item_id)
                )
        return {"status": "success", "data": {"id": int(item_id), "version": next_ver}}

    if visibility not in ("private", "group"):
        raise HTTPException(status_code=400, detail="Visibility must be 'private' or 'group'")
    if visibility == "group":
        if not group_id:
            raise HTTPException(status_code=400, detail="group_id is required for group visibility")
        group_id = int(group_id)
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        if group_id not in group_roles:
            raise HTTPException(status_code=403, detail="Not a member of the group")
    else:
        group_id = None

    if parent_id is not None:
        parent = await get_cloud_item_for_user(request, int(parent_id), user)
        if parent.get('item_type') != 'folder':
            raise HTTPException(status_code=400, detail="Parent must be a folder")

    storage_path = cloud_storage_path(user["id"], filename)
    meta = save_upload_stream(file, storage_path)
    if not ensure_within_storage(storage_path):
        raise HTTPException(status_code=400, detail="Invalid storage path")

    tags_list = normalize_tags(tags)

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO cloud_items (owner_user_id, parent_id, name, item_type, size, mime_type, storage_path,
                                         checksum, visibility, group_id, tags)
                VALUES (%s, %s, %s, 'file', %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user["id"], parent_id, filename, meta["size"], mime_type, storage_path, meta["checksum"], visibility, group_id, tags_list)
            )
            row = await cur.fetchone()
            item_id = row["id"]
            await cur.execute(
                """
                INSERT INTO cloud_versions (item_id, version_num, size, checksum, storage_path, created_by)
                VALUES (%s, 1, %s, %s, %s, %s)
                """,
                (item_id, meta["size"], meta["checksum"], storage_path, user["id"])
            )

    return {"status": "success", "data": {"id": item_id}}

@app.put("/api/v1/cloud/items/{item_id}")
async def api_v1_cloud_update_item(
    request: Request,
    item_id: int,
    data: dict,
    user: dict = Depends(require_scope('cloud'))
):
    name = data.get('name')
    parent_id = data.get('parent_id')
    is_archived = data.get('is_archived')
    is_favorited = data.get('is_favorited')
    tags = data.get('tags')
    is_shared = data.get('is_shared')
    visibility = data.get('visibility')
    group_id = data.get('group_id')

    item = await get_cloud_item_for_user(request, item_id, user)

    updates = []
    params = []

    if name is not None:
        updates.append("name = %s")
        params.append(safe_filename(str(name)))
    if 'parent_id' in data:
        if parent_id is None:
            updates.append("parent_id = NULL")
        else:
            if parent_id == item_id:
                raise HTTPException(status_code=400, detail="Invalid parent_id")
            parent = await get_cloud_item_for_user(request, int(parent_id), user)
            if parent.get('item_type') != 'folder':
                raise HTTPException(status_code=400, detail="Parent must be a folder")
            updates.append("parent_id = %s")
            params.append(parent_id)
    if isinstance(is_archived, bool):
        updates.append("is_archived = %s")
        params.append(is_archived)
    if isinstance(is_favorited, bool):
        updates.append("is_favorited = %s")
        params.append(is_favorited)
    if isinstance(is_shared, bool):
        updates.append("is_shared = %s")
        params.append(is_shared)
    if tags is not None:
        updates.append("tags = %s")
        params.append(normalize_tags(tags))
    if visibility is not None:
        if visibility not in ("private", "group"):
            raise HTTPException(status_code=400, detail="Visibility must be 'private' or 'group'")
        if visibility == "group":
            if not group_id:
                raise HTTPException(status_code=400, detail="group_id is required for group visibility")
            group_id = int(group_id)
            group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
            if group_id not in group_roles:
                raise HTTPException(status_code=403, detail="Not a member of the group")
        else:
            group_id = None
        updates.append("visibility = %s")
        params.append(visibility)
        updates.append("group_id = %s")
        params.append(group_id)

    if not updates:
        return {"status": "success"}

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(item_id)

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE cloud_items SET {', '.join(updates)} WHERE id = %s",
                params
            )

    return {"status": "success"}

@app.delete("/api/v1/cloud/items/{item_id}")
async def api_v1_cloud_delete_item(
    request: Request,
    item_id: int,
    user: dict = Depends(require_scope('cloud'))
):
    async def _delete_recursive(conn, target_id: int):
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, item_type, storage_path FROM cloud_items WHERE id = %s",
                (target_id,)
            )
            row = await cur.fetchone()
            if not row:
                return
            await cur.execute(
                "SELECT id FROM cloud_items WHERE parent_id = %s",
                (target_id,)
            )
            children = await cur.fetchall()
        for child in children:
            await _delete_recursive(conn, child["id"])
        if row["item_type"] == 'file' and row.get("storage_path"):
            try:
                if ensure_within_storage(row["storage_path"]) and os.path.exists(row["storage_path"]):
                    os.remove(row["storage_path"])
            except Exception:
                pass
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM cloud_items WHERE id = %s", (target_id,))

    await get_cloud_item_for_user(request, item_id, user)
    async with request.app.state.db_pool.connection() as conn:
        await _delete_recursive(conn, item_id)
    return {"status": "success"}

@app.get("/api/v1/cloud/items/{item_id}/download")
async def api_v1_cloud_download_item(
    request: Request,
    item_id: int,
    inline: bool = False,
    user: dict = Depends(require_scope('cloud'))
):
    item = await get_cloud_item_for_user(request, item_id, user)
    if item.get('item_type') != 'file':
        raise HTTPException(status_code=400, detail="Only files can be downloaded")
    storage_path = item.get('storage_path')
    if not storage_path or not os.path.exists(storage_path) or not ensure_within_storage(storage_path):
        raise HTTPException(status_code=404, detail="File not found")
    disposition = "inline" if inline else "attachment"
    headers = {"Content-Disposition": f"{disposition}; filename=\"{item['name']}\""}
    return FileResponse(storage_path, media_type=item.get('mime_type') or 'application/octet-stream', headers=headers)

@app.get("/api/v1/cloud/items/{item_id}/versions")
async def api_v1_cloud_list_versions(
    request: Request,
    item_id: int,
    user: dict = Depends(require_scope('cloud'))
):
    await get_cloud_item_for_user(request, item_id, user)
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, version_num, size, checksum, created_at
                FROM cloud_versions
                WHERE item_id = %s
                ORDER BY version_num DESC
                """,
                (item_id,)
            )
            rows = await cur.fetchall()
    results = [dict(row) for row in rows]
    for r in results:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].isoformat()
    return {"status": "success", "data": results}

@app.get("/api/v1/cloud/versions/{version_id}/download")
async def api_v1_cloud_download_version(
    request: Request,
    version_id: int,
    user: dict = Depends(require_scope('cloud'))
):
    group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT cv.storage_path, cv.size, ci.name, ci.mime_type
                FROM cloud_versions cv
                JOIN cloud_items ci ON ci.id = cv.item_id
                WHERE cv.id = %s
                  AND (ci.owner_user_id = %s OR (ci.visibility = 'group' AND ci.group_id = ANY(%s)))
                """,
                (version_id, user["id"], list(group_roles.keys()))
            )
            row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Version not found")
    storage_path = row["storage_path"]
    if not storage_path or not os.path.exists(storage_path) or not ensure_within_storage(storage_path):
        raise HTTPException(status_code=404, detail="File not found")
    headers = {"Content-Disposition": f"attachment; filename=\"{row['name']}\""}
    return FileResponse(storage_path, media_type=row.get('mime_type') or 'application/octet-stream', headers=headers)

@app.post("/api/v1/cloud/items/{item_id}/share")
async def api_v1_cloud_share_item(
    request: Request,
    item_id: int,
    data: dict,
    user: dict = Depends(require_scope('cloud'))
):
    item = await get_cloud_item_for_user(request, item_id, user)
    if item.get('item_type') != 'file':
        raise HTTPException(status_code=400, detail="Only files can be shared")
    expires_in_hours = int(data.get('expires_in_hours') or 0)
    max_uses = int(data.get('max_uses') or 0)
    expires_at = None
    if expires_in_hours > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    token = secrets.token_urlsafe(24)

    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO cloud_shares (token, item_id, created_by, expires_at, max_uses)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (token, item_id, user["id"], expires_at, max_uses)
            )
            await cur.execute("UPDATE cloud_items SET is_shared = TRUE WHERE id = %s", (item_id,))

    return {"status": "success", "data": {"share_token": token, "share_url": f"/api/v1/cloud/share/{token}"}}

@app.get("/api/v1/cloud/share/{token}")
async def api_v1_cloud_download_shared(token: str):
    async with app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT cs.token, cs.expires_at, cs.max_uses, cs.uses,
                       ci.name, ci.storage_path, ci.mime_type, ci.item_type
                FROM cloud_shares cs
                JOIN cloud_items ci ON ci.id = cs.item_id
                WHERE cs.token = %s
                """,
                (token,)
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Share not found")

            if row["expires_at"] and row["expires_at"] < datetime.now(timezone.utc):
                raise HTTPException(status_code=410, detail="Share expired")
            if row["max_uses"] and row["uses"] >= row["max_uses"]:
                raise HTTPException(status_code=410, detail="Share usage exceeded")
            if row["item_type"] != 'file':
                raise HTTPException(status_code=400, detail="Shared item is not a file")

            await cur.execute("UPDATE cloud_shares SET uses = uses + 1 WHERE token = %s", (token,))

    storage_path = row["storage_path"]
    if not storage_path or not os.path.exists(storage_path) or not ensure_within_storage(storage_path):
        raise HTTPException(status_code=404, detail="File not found")
    headers = {"Content-Disposition": f"attachment; filename=\"{row['name']}\""}
    return FileResponse(storage_path, media_type=row.get('mime_type') or 'application/octet-stream', headers=headers)

@app.post("/api/v1/cloud/ocr")
async def api_v1_cloud_ocr(
    request: Request,
    data: dict,
    user: dict = Depends(require_scope('cloud'))
):
    image_base64 = data.get('image_base64')
    file_id = data.get('file_id')

    if file_id:
        item = await get_cloud_item_for_user(request, int(file_id), user)
        if not item.get('mime_type', '').startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files can be OCR processed")
        storage_path = item.get('storage_path')
        if not storage_path or not os.path.exists(storage_path) or not ensure_within_storage(storage_path):
            raise HTTPException(status_code=404, detail="File not found")
        with open(storage_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("ascii")

    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 or file_id is required")

    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[-1]

    text = await ocr_with_ai(image_base64)
    return {"status": "success", "data": {"text": text}}

# --- 管理员后台 ---

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/v1/admin/users")
async def admin_list_users(request: Request, admin: dict = Depends(require_admin())):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, email, name, role, created_at,
                (SELECT COUNT(*) FROM cloud_items WHERE owner_user_id = users.id) as file_count,
                (SELECT SUM(size) FROM cloud_items WHERE owner_user_id = users.id) as storage_used
                FROM users ORDER BY created_at DESC
            """)
            users = await cur.fetchall()
            return {"users": users}

@app.get("/api/v1/admin/settings")
async def admin_get_settings(request: Request, admin: dict = Depends(require_admin())):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT key, value FROM system_settings")
            rows = await cur.fetchall()
            settings = {row["key"]: row["value"] for row in rows}
            return {"settings": settings}

@app.patch("/api/v1/admin/settings")
async def admin_update_settings(request: Request, data: dict, admin: dict = Depends(require_admin())):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor() as cur:
            for key, value in data.items():
                await cur.execute("""
                    INSERT INTO system_settings (key, value) VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """, (key, json.dumps(value)))
            return {"message": "Settings updated"}

@app.get("/api/v1/admin/stats")
async def admin_stats(request: Request, admin: dict = Depends(require_admin())):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT COUNT(*) as total_users FROM users")
            user_stats = await cur.fetchone()
            
            await cur.execute("SELECT COUNT(*) as total_files, SUM(size) as total_size FROM cloud_items")
            file_stats = await cur.fetchone()
            
            await cur.execute("SELECT COUNT(*) as total_reads FROM reading_list")
            reading_stats = await cur.fetchone()

            return {
                "users": user_stats["total_users"],
                "files": file_stats["total_files"] or 0,
                "storage_used": int(file_stats["total_size"] or 0),
                "readings": reading_stats["total_reads"] or 0
            }

@app.patch("/api/v1/admin/users/{user_id}")
async def admin_update_user(request: Request, user_id: int, data: dict, admin: dict = Depends(require_admin())):
    allowed_fields = ["role", "name"]
    updates = []
    params = []
    
    for field in allowed_fields:
        if field in data:
            updates.append(f"{field} = %s")
            params.append(data[field])
            
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")
        
    params.append(user_id)
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", params)
            return {"message": "User updated"}

# --- 5. API密钥管理 ---

def generate_api_key() -> tuple[str, str, str]:
    """生成API密钥，返回 (完整密钥, 密钥hash, 前缀)"""
    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:8]
    return raw_key, key_hash, key_prefix

@app.get("/api/keys")
async def api_list_keys(request: Request, user: dict = Depends(get_current_user)):
    """列出用户的所有API密钥（不显示完整密钥）"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, key_name, key_prefix, scopes, last_used, created_at
                FROM user_api_keys
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user["id"],)
            )
            keys = await cur.fetchall()
    
    result = []
    for k in keys:
        result.append({
            "id": k["id"],
            "name": k["key_name"],
            "prefix": k["key_prefix"],
            "scopes": k["scopes"].split(',') if k["scopes"] else [],
            "last_used": k["last_used"].isoformat() if k["last_used"] else None,
            "created_at": k["created_at"].isoformat()
        })
    return {"keys": result}

@app.post("/api/keys")
async def api_create_key(request: Request, data: dict, user: dict = Depends(get_current_user)):
    """创建新的API密钥"""
    key_name = (data.get("name") or "").strip()
    scopes_list = data.get("scopes", ["read", "write", "reading", "cloud"])  # 默认读写 + 阅读/云盘
    
    if not key_name:
        raise HTTPException(status_code=400, detail="Key name is required")
    
    # 验证scopes
    valid_scopes = {"read", "write", "admin", "reading", "cloud"}
    if not isinstance(scopes_list, list):
        raise HTTPException(status_code=400, detail="Scopes must be an array")
    
    for scope in scopes_list:
        if scope not in valid_scopes:
            raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
    
    scopes_str = ','.join(scopes_list)
    raw_key, key_hash, key_prefix = generate_api_key()
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO user_api_keys (user_id, key_name, key_hash, key_prefix, scopes)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user["id"], key_name, key_hash, key_prefix, scopes_str)
            )
            key_record = await cur.fetchone()
    
    return {
        "status": "success",
        "key": raw_key,
        "key_id": key_record["id"],
        "scopes": scopes_list,
        "message": "请妥善保存此密钥，它只会显示一次"
    }

@app.delete("/api/keys/{key_id}")
async def api_delete_key(request: Request, key_id: int, user: dict = Depends(get_current_user)):
    """删除API密钥"""
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id FROM user_api_keys WHERE id = %s AND user_id = %s",
                (key_id, user["id"])
            )
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="Key not found")
            
            await cur.execute(
                "DELETE FROM user_api_keys WHERE id = %s",
                (key_id,)
            )
    
    return {"status": "success"}

# --- 6. 个人资料和安全设置 ---

@app.get("/api/profile")
async def api_get_profile(user: dict = Depends(get_current_user)):
    """获取用户个人资料"""
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"]
    }

@app.put("/api/profile")
async def api_update_profile(request: Request, data: dict, user: dict = Depends(get_current_user)):
    """更新用户个人资料"""
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # 检查邮箱是否被其他用户占用
            if email != user["email"]:
                await cur.execute(
                    "SELECT id FROM users WHERE email = %s AND id != %s",
                    (email, user["id"])
                )
                if await cur.fetchone():
                    raise HTTPException(status_code=409, detail="Email already in use")
            
            await cur.execute(
                "UPDATE users SET name = %s, email = %s WHERE id = %s",
                (name or None, email, user["id"])
            )
    
    return {"status": "success", "message": "个人资料已更新"}

@app.put("/api/security/password")
async def api_change_password(request: Request, data: dict, user: dict = Depends(get_current_user)):
    """修改密码"""
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")
    
    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail="Current and new passwords are required")
    
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT password_hash, password_salt FROM users WHERE id = %s",
                (user["id"],)
            )
            user_data = await cur.fetchone()
            
            if not verify_password(current_password, user_data["password_salt"], user_data["password_hash"]):
                raise HTTPException(status_code=401, detail="Current password is incorrect")
            
            pw = hash_password(new_password)
            await cur.execute(
                "UPDATE users SET password_hash = %s, password_salt = %s WHERE id = %s",
                (pw["hash"], pw["salt"], user["id"])
            )
    
    return {"status": "success", "message": "密码已成功修改"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
