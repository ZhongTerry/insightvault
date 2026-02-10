import requests
from bs4 import BeautifulSoup
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
conn_info = f"host={os.getenv('DB_HOST', 'localhost')} port={os.getenv('DB_PORT', '5432')} dbname={os.getenv('DB_NAME', 'neuro_note')} user={os.getenv('DB_USER', 'postgres')} password={os.getenv('DB_PASSWORD', 'password')}"

async def get_db_connection():
    try:
        conn = await psycopg.AsyncConnection.connect(conn_info, autocommit=True)
        return conn
    except Exception as e:
        print(f"数据库连接失败: {e}")
        sys.exit(1)

async def get_default_user(conn):
    """获取第一个用户作为默认归属者"""
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT id, email FROM users ORDER BY id ASC LIMIT 1")
        user = await cur.fetchone()
        if not user:
            print("错误: 数据库中没有任何用户，无法归档情报。请先注册用户。")
            sys.exit(1)
        return user

async def ensure_tag(conn, user_id, tag_name):
    """确保标签存在并返回ID"""
    async with conn.cursor(row_factory=dict_row) as cur:
        # 查找私有标签
        await cur.execute(
            "SELECT id FROM tags WHERE visibility = 'private' AND owner_user_id = %s AND name = %s",
            (user_id, tag_name)
        )
        row = await cur.fetchone()
        if row:
            return row['id']
        
        # 创建新标签
        await cur.execute(
            "INSERT INTO tags (name, owner_user_id, visibility) VALUES (%s, %s, 'private') RETURNING id",
            (tag_name, user_id)
        )
        row = await cur.fetchone()
        return row['id']

async def add_item_tags(conn, item_id, tag_ids):
    async with conn.cursor() as cur:
        for tag_id in tag_ids:
            # 避免重复插入
            await cur.execute(
                "INSERT INTO item_tags (item_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (item_id, tag_id)
            )

async def save_to_vault(conn, user_id, news_item):
    """保存新闻到情报库"""
    category = news_item['cate']
    title = news_item['title']
    url = news_item['url']
    
    # 构建内容
    content = f"### {title}\n\n**来源**: [{category} - 网易新闻]({url})\n\n*(此条目由爬虫自动采集)*"

    async with conn.cursor(row_factory=dict_row) as cur:
        # 简单查重：检查 content 中是否包含该 URL
        await cur.execute(
            "SELECT id FROM intelligence_vault WHERE owner_user_id = %s AND content LIKE %s LIMIT 1",
            (user_id, f"%{url}%")
        )
        existing = await cur.fetchone()
        if existing:
            print(f"[-] 跳过已存在: {title}")
            return False

        # 插入情报
        await cur.execute(
            """
            INSERT INTO intelligence_vault (title, content, owner_user_id, visibility, vectorization_status)
            VALUES (%s, %s, %s, 'private', 'pending')
            RETURNING id
            """,
            (title, content, user_id)
        )
        row = await cur.fetchone()
        item_id = row['id']
        
        #处理标签
        tag_netease = await ensure_tag(conn, user_id, "网易新闻")
        tag_cate = await ensure_tag(conn, user_id, category)
        
        await add_item_tags(conn, item_id, [tag_netease, tag_cate])
        print(f"[+] 已保存: {title} (ID: {item_id})")
        return True

def crawl_netease_news():
    """该函数包含原始爬虫逻辑"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    final_report = []

    print("正在爬取网易新闻...")

    # --- 第一部分：抓取国际要闻 (World) ---
    world_url = "https://news.163.com/world/"
    try:
        resp_world = requests.get(world_url, headers=headers, timeout=10)
        resp_world.encoding = resp_world.apparent_encoding
        soup_world = BeautifulSoup(resp_world.text, 'html.parser')
        
        world_box = soup_world.find('div', class_='hidden')
        if world_box:
            links = world_box.find_all('a')
            for link in links[:10]: 
                final_report.append({
                    "cate": "国际政治",
                    "title": link.get_text(strip=True),
                    "url": link.get('href')
                })
    except Exception as e:
        print(f"国际频道抓取失败: {e}")

    # --- 第二部分：抓取科技快讯 (Tech) ---
    tech_url = "https://tech.163.com/"
    try:
        resp_tech = requests.get(tech_url, headers=headers, timeout=10)
        resp_tech.encoding = resp_tech.apparent_encoding
        soup_tech = BeautifulSoup(resp_tech.text, 'html.parser')
        
        tech_list = soup_tech.find('div', class_='newest-lists')
        if tech_list:
            items = tech_list.find_all('li', class_='list_item')
            for item in items[:10]: 
                a_tag = item.find('a', class_='nl_detail')
                title_p = item.find('p', class_='nl-title')
                
                if a_tag and title_p:
                    if title_p.find('em'):
                        title_p.find('em').decompose()
                    
                    final_report.append({
                        "cate": "科技趋势",
                        "title": title_p.get_text(strip=True),
                        "url": a_tag.get('href')
                    })
    except Exception as e:
        print(f"科技频道抓取失败: {e}")

    return final_report

async def main():
    # 1. 爬取数据
    news_items = crawl_netease_news()
    if not news_items:
        print("未获取到任何新闻数据。")
        return

    # 2. 连接数据库
    conn = await get_db_connection()
    try:
        # 3. 获取归属用户
        user = await get_default_user(conn)
        print(f"正在为用户 {user['email']} (ID: {user['id']}) 归档情报...")

        # 4. 存入数据库
        count = 0
        for item in news_items:
            success = await save_to_vault(conn, user['id'], item)
            if success:
                count += 1
        
        print(f"\n处理完成！共新增 {count} 条情报。")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
