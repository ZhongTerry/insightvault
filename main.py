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
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

# 解决 Windows 下 Psycopg 异步连接池与默认事件循环 (ProactorEventLoop) 的兼容性问题
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import httpx
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
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
            await conn.execute("UPDATE intelligence_vault SET visibility = 'private' WHERE visibility IS NULL;")
            await conn.execute("UPDATE intelligence_vault SET vectorization_status = 'success' WHERE vectorization_status IS NULL AND embedding IS NOT NULL;")
            await conn.execute("UPDATE intelligence_vault SET vectorization_status = 'pending' WHERE vectorization_status IS NULL AND embedding IS NULL;")

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
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_owner ON reading_list (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_created ON reading_list (created_at DESC);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_status ON reading_list (owner_user_id, is_read, is_archived);")
            
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
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
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
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    
    token = auth.split(" ", 1)[1].strip()
    
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

PER_PAGE = 10

async def fetch_data(
    pool: AsyncConnectionPool,
    user_id: int,
    group_ids: List[int],
    search_type: str = 'all',
    query: str = '',
    page: int = 1
):
    offset = (page - 1) * PER_PAGE
    results = []
    
    # 步骤 A: 获取向量
    query_vec = None
    if search_type == 'ai' and query:
        query_vec = await get_embedding(query)
        if query_vec:
            logger.info(f"AI 搜索向量维度: {len(query_vec)}")

    # 步骤 B: 数据库操作
    with Profiler() as p_db:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                access_sql = "(iv.owner_user_id = %s OR (iv.visibility = 'group' AND iv.group_id = ANY(%s)))"
                if search_type == 'ai' and query_vec:
                    # 显式使用 [..]::vector 进行类型转换，确保 pgvector 正确识别
                    await cur.execute("""
                        SELECT iv.id, iv.title, iv.content, iv.created_at, iv.visibility, iv.group_id,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name
                                   FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags,
                               1 - (iv.embedding <=> %s::vector) AS score
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE """ + access_sql + """
                        ORDER BY score DESC LIMIT %s OFFSET %s
                    """, (json.dumps(query_vec), user_id, group_ids, PER_PAGE, offset))
                elif search_type == 'key' and query:
                    await cur.execute("""
                        SELECT iv.id, iv.title, iv.content, iv.created_at, iv.visibility, iv.group_id,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name
                                   FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE (iv.content ILIKE %s OR iv.title ILIKE %s) AND """ + access_sql + """
                        ORDER BY iv.created_at DESC LIMIT %s OFFSET %s
                    """, (f'%{query}%', f'%{query}%', user_id, group_ids, PER_PAGE, offset))
                elif search_type == 'tag' and query:
                    tag = query.strip().lower()
                    await cur.execute("""
                        SELECT iv.id, iv.title, iv.content, iv.created_at, iv.visibility, iv.group_id,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name
                                   FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE """ + access_sql + """ AND EXISTS (
                            SELECT 1
                            FROM item_tags it
                            JOIN tags t ON t.id = it.tag_id
                            WHERE it.item_id = iv.id AND t.name = %s
                        )
                        ORDER BY iv.created_at DESC LIMIT %s OFFSET %s
                    """, (user_id, group_ids, tag, PER_PAGE, offset))
                else:
                    await cur.execute("""
                        SELECT iv.id, iv.title, iv.content, iv.created_at, iv.visibility, iv.group_id,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name
                                   FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE """ + access_sql + """
                        ORDER BY iv.created_at DESC LIMIT %s OFFSET %s
                    """, (user_id, group_ids, PER_PAGE, offset))
                
                rows = await cur.fetchall()
                # 显式转换为 dict 列表
                results = [dict(row) for row in rows]
    
    logger.info(f"SQL 查询完成 | 模式: {search_type} | 耗时: {p_db.duration:.2f}ms | 数量: {len(results)}")
    
    # 时间格式化
    for r in results:
        if isinstance(r['created_at'], datetime):
            r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M')
    return results

# --- 4. 鉴权与用户组 ---

@app.post("/api/auth/register")
async def api_register(request: Request, data: dict):
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

@app.get("/api/search")
async def api_search(request: Request, type: str = 'all', q: str = '', page: int = 1, user: dict = Depends(get_current_user)):
    with Profiler() as p_total:
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        results = await fetch_data(request.app.state.db_pool, user["id"], list(group_roles.keys()), type, q, page)
    logger.info(f">>> 搜索请求总响应时间: {p_total.duration:.2f}ms")
    return results

@app.post("/api/add")
async def api_add(request: Request, background_tasks: BackgroundTasks, data: dict, user: dict = Depends(get_current_user)):
    with Profiler() as p_total:
        content = data.get('content')
        visibility = (data.get("visibility") or "private").strip()
        group_id = data.get("group_id")
        tags_input = data.get("tags")
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        if visibility not in ("private", "group"):
            raise HTTPException(status_code=400, detail="Visibility must be private or group")

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
        
        with Profiler() as p_db:
            # 先插入数据（embedding 为 NULL）
            async with request.app.state.db_pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        """
                        INSERT INTO intelligence_vault (content, embedding, owner_user_id, visibility, group_id)
                        VALUES (%s, NULL, %s, %s, %s)
                        RETURNING id
                        """,
                        (content, user["id"], visibility, group_id)
                    )
                    row = await cur.fetchone()
                    item_id = row["id"]

            await set_item_tags(request.app.state.db_pool, item_id, user["id"], visibility, group_id, tags)
        
        # 启动后台向量化任务
        background_tasks.add_task(background_vectorize_item, request.app.state.db_pool, item_id, content)
        
        logger.info(f"DB 写入完成 | 耗时: {p_db.duration:.2f}ms")
    logger.info(f">>> 录入请求总响应时间: {p_total.duration:.2f}ms")
    return {"status": "success"}

@app.delete("/api/delete/{item_id}")
async def api_delete(request: Request, item_id: int, user: dict = Depends(get_current_user)):
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, owner_user_id, visibility, group_id FROM intelligence_vault WHERE id = %s",
                (item_id,)
            )
            item = await cur.fetchone()
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")

            if item["owner_user_id"] != user["id"]:
                allowed = False
                if item["visibility"] == "group" and item["group_id"]:
                    allowed = await is_group_admin(request.app.state.db_pool, user["id"], item["group_id"])
                if not allowed:
                    raise HTTPException(status_code=403, detail="No permission to delete")

            await cur.execute("DELETE FROM intelligence_vault WHERE id = %s", (item_id,))
    return {"status": "success"}

@app.put("/api/update/{item_id}")
async def api_update(request: Request, background_tasks: BackgroundTasks, item_id: int, data: dict, user: dict = Depends(get_current_user)):
    new_content = data.get('content')
    visibility = (data.get("visibility") or "private").strip()
    group_id = data.get("group_id")
    tags_input = data.get("tags", None)

    if not new_content:
        raise HTTPException(status_code=400, detail="Content is required")
    if visibility not in ("private", "group"):
        raise HTTPException(status_code=400, detail="Visibility must be private or group")

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

            # 先更新数据（不更新 embedding）
            await cur.execute(
                """
                UPDATE intelligence_vault
                SET content = %s, visibility = %s, group_id = %s
                WHERE id = %s
                """,
                (new_content, visibility, group_id, item_id)
            )

    if tags_input is not None:
        tags = normalize_tags(tags_input)
        await set_item_tags(request.app.state.db_pool, item_id, user["id"], visibility, group_id, tags)
    
    # 启动后台向量化任务
    background_tasks.add_task(background_vectorize_item, request.app.state.db_pool, item_id, new_content)
    
    return {"status": "success"}

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

@app.post("/api/revectorize_all")
async def api_revectorize_all(request: Request, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    await require_admin(user)
    # 立即返回，后台执行
    background_tasks.add_task(background_revectorize, request.app.state.db_pool)
    return {"status": "success", "message": "Task started in background"}

# --- REST API v1 (新版API with proper REST conventions) ---

@app.get("/api/v1/items")
async def api_v1_list_items(
    request: Request,
    type: str = 'all',
    q: str = '',
    page: int = 1,
    per_page: int = Query(default=10, le=100),
    user: dict = Depends(require_scope('read'))
):
    """列出情报项目（支持搜索和分页）"""
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
                
                if type == 'ai' and query_vec:
                    await cur.execute("""
                        SELECT iv.id, iv.title, 
                               LEFT(iv.content, 200) as content_preview,
                               iv.created_at, iv.visibility, iv.group_id,
                               iv.vectorization_status,
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
                        WHERE """ + access_sql + """
                        ORDER BY score DESC LIMIT %s OFFSET %s
                    """, (json.dumps(query_vec), user["id"], list(group_roles.keys()), per_page, offset))
                elif type == 'key' and q:
                    await cur.execute("""
                        SELECT iv.id, iv.title,
                               LEFT(iv.content, 200) as content_preview,
                               iv.created_at, iv.visibility, iv.group_id,
                               iv.vectorization_status,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE (iv.content ILIKE %s OR iv.title ILIKE %s) AND """ + access_sql + """
                        ORDER BY iv.created_at DESC LIMIT %s OFFSET %s
                    """, (f'%{q}%', f'%{q}%', user["id"], list(group_roles.keys()), per_page, offset))
                else:
                    await cur.execute("""
                        SELECT iv.id, iv.title,
                               LEFT(iv.content, 200) as content_preview,
                               iv.created_at, iv.visibility, iv.group_id,
                               iv.vectorization_status,
                               g.name AS group_name,
                               ARRAY(
                                   SELECT t.name FROM item_tags it
                                   JOIN tags t ON t.id = it.tag_id
                                   WHERE it.item_id = iv.id
                                   ORDER BY t.name
                               ) AS tags
                        FROM intelligence_vault iv
                        LEFT JOIN groups g ON g.id = iv.group_id
                        WHERE """ + access_sql + """
                        ORDER BY iv.created_at DESC LIMIT %s OFFSET %s
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
                       iv.vectorization_status,
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

# --- Reading List API (稍后阅读) ---

@app.post("/api/v1/reading-list")
async def api_v1_create_reading_item(
    request: Request,
    data: dict,
    user: dict = Depends(require_scope('write'))
):
    """添加稍后阅读项目"""
    title = (data.get('title') or '').strip()
    url = (data.get('url') or '').strip()
    content = (data.get('content') or '').strip()
    source = (data.get('source') or '').strip()
    cover_image = (data.get('cover_image') or '').strip()
    
    if not title or not url:
        raise HTTPException(status_code=400, detail="Title and URL are required")
    
    has_content = bool(content)
    
    async with request.app.state.db_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO reading_list (title, url, content, source, cover_image, has_content, owner_user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (title, url, content or None, source or None, cover_image or None, has_content, user["id"])
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
    user: dict = Depends(require_scope('read'))
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
                SELECT id, title, url, source, cover_image, has_content, is_read, is_archived, created_at
                {base_query}
                ORDER BY created_at DESC
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
    user: dict = Depends(require_scope('read'))
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
    user: dict = Depends(require_scope('write'))
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
    user: dict = Depends(require_scope('write'))
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
            
            # 如果没有内容，不允许保存
            if not reading_item['has_content']:
                raise HTTPException(status_code=400, detail="Cannot save item without content")
            
            # 创建情报项
            title = f"{reading_item['title']} (来源: {reading_item['source'] or reading_item['url']})"
            content = f"# {reading_item['title']}\n\n**来源**: {reading_item['url']}\n\n---\n\n{reading_item['content']}"
            
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
    scopes_list = data.get("scopes", ["read", "write"])  # 默认读写权限
    
    if not key_name:
        raise HTTPException(status_code=400, detail="Key name is required")
    
    # 验证scopes
    valid_scopes = {"read", "write", "admin"}
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
