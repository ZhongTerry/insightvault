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
                    content TEXT NOT NULL,
                    embedding vector({detected_dim}),
                    owner_user_id INTEGER REFERENCES users(id),
                    visibility TEXT DEFAULT 'private',
                    group_id INTEGER REFERENCES groups(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS owner_user_id INTEGER REFERENCES users(id);")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS visibility TEXT DEFAULT 'private';")
            await conn.execute("ALTER TABLE intelligence_vault ADD COLUMN IF NOT EXISTS group_id INTEGER REFERENCES groups(id);")
            await conn.execute("UPDATE intelligence_vault SET visibility = 'private' WHERE visibility IS NULL;")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_iv_owner ON intelligence_vault (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_iv_group ON intelligence_vault (group_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_iv_hnsw ON intelligence_vault USING hnsw (embedding vector_cosine_ops);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_owner ON tags (owner_user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_group ON tags (group_id);")
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_tags_private ON tags (owner_user_id, name) WHERE visibility = 'private';")
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_tags_group ON tags (group_id, name) WHERE visibility = 'group';")
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
                        SELECT iv.id, iv.content, iv.created_at, iv.visibility, iv.group_id,
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
                        SELECT iv.id, iv.content, iv.created_at, iv.visibility, iv.group_id,
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
                        WHERE iv.content ILIKE %s AND """ + access_sql + """
                        ORDER BY iv.created_at DESC LIMIT %s OFFSET %s
                    """, (f'%{query}%', user_id, group_ids, PER_PAGE, offset))
                elif search_type == 'tag' and query:
                    tag = query.strip().lower()
                    await cur.execute("""
                        SELECT iv.id, iv.content, iv.created_at, iv.visibility, iv.group_id,
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
                        SELECT iv.id, iv.content, iv.created_at, iv.visibility, iv.group_id,
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

@app.get("/api/search")
async def api_search(request: Request, type: str = 'all', q: str = '', page: int = 1, user: dict = Depends(get_current_user)):
    with Profiler() as p_total:
        group_roles = await get_user_group_roles(request.app.state.db_pool, user["id"])
        results = await fetch_data(request.app.state.db_pool, user["id"], list(group_roles.keys()), type, q, page)
    logger.info(f">>> 搜索请求总响应时间: {p_total.duration:.2f}ms")
    return results

@app.post("/api/add")
async def api_add(request: Request, data: dict, user: dict = Depends(get_current_user)):
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
        
        vec = await get_embedding(content)
        tags = normalize_tags(tags_input)
        
        with Profiler() as p_db:
            async with request.app.state.db_pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        """
                        INSERT INTO intelligence_vault (content, embedding, owner_user_id, visibility, group_id)
                        VALUES (%s, %s::vector, %s, %s, %s)
                        RETURNING id
                        """,
                        (content, json.dumps(vec), user["id"], visibility, group_id)
                    )
                    row = await cur.fetchone()
                    item_id = row["id"]

            await set_item_tags(request.app.state.db_pool, item_id, user["id"], visibility, group_id, tags)
        
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
async def api_update(request: Request, item_id: int, data: dict, user: dict = Depends(get_current_user)):
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

            new_vec = await get_embedding(new_content)
            await cur.execute(
                """
                UPDATE intelligence_vault
                SET content = %s, embedding = %s::vector, visibility = %s, group_id = %s
                WHERE id = %s
                """,
                (new_content, json.dumps(new_vec), visibility, group_id, item_id)
            )

    if tags_input is not None:
        tags = normalize_tags(tags_input)
        await set_item_tags(request.app.state.db_pool, item_id, user["id"], visibility, group_id, tags)
    return {"status": "success"}

# 异步背景任务：全量重新向量化
async def background_revectorize(pool: AsyncConnectionPool):
    logger.info("开始执行异步全量重新向量化任务...")
    start_time = time.time()
    try:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT id, content FROM intelligence_vault")
                all_items = await cur.fetchall()
                
                count = 0
                for item in all_items:
                    # 强制刷新向量 (这里不使用缓存逻辑，因为是"重构")
                    base_url = os.getenv("AI_BASE_URL").rstrip('/')
                    endpoint = f"{base_url}/embeddings"
                    headers = {"Authorization": f"Bearer {os.getenv('AI_API_KEY')}"}
                    payload = {"model": os.getenv("EMBEDDING_MODEL"), "input": item['content']}
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(endpoint, json=payload, headers=headers, timeout=10.0)
                        if response.status_code == 200:
                            vec = response.json()['data'][0]['embedding']
                            # 再次强调：使用 ::vector 确保类型安全
                            await conn.execute(
                                "UPDATE intelligence_vault SET embedding = %s::vector WHERE id = %s",
                                (json.dumps(vec), item['id'])
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
