import os
import asyncio
import logging
import httpx
import sys
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv

# 解决 Windows 下 Psycopg 异步连接池与默认事件循环 (ProactorEventLoop) 的兼容性问题
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("VectorizeTags")

load_dotenv()

# 数据库配置
conn_info = f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')}"

async def get_embedding(text: str):
    base_url = os.getenv("AI_BASE_URL").rstrip('/')
    endpoint = f"{base_url}/embeddings"
    headers = {"Authorization": f"Bearer {os.getenv('AI_API_KEY')}"}
    payload = {"model": os.getenv("EMBEDDING_MODEL"), "input": text}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, json=payload, headers=headers, timeout=20.0)
        response.raise_for_status()
        return response.json()['data'][0]['embedding']

async def main():
    async with AsyncConnectionPool(conninfo=conn_info, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # 0. 检查并添加 embedding 列
                logger.info("正在检测模型维度并同步数据库列...")
                test_vec = await get_embedding("DimCheck")
                dim = len(test_vec)
                
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cur.execute(f"ALTER TABLE system_tags ADD COLUMN IF NOT EXISTS embedding vector({dim})")
                
                # 1. 获取所有没有向量的标签
                await cur.execute("SELECT name, description FROM system_tags WHERE embedding IS NULL")
                tags = await cur.fetchall()
                
                if not tags:
                    logger.info("所有系统标签都已向量化，无需操作。")
                    return

                logger.info(f"开启向量化任务，共 {len(tags)} 个标签待处理...")
                
                success_count = 0
                for tag in tags:
                    name = tag['name']
                    description = tag['description']
                    text = f"{name}: {description}"
                    
                    try:
                        logger.info(f"正在处理标签: {name} ...")
                        vec = await get_embedding(text)
                        
                        # 2. 更新到数据库
                        await cur.execute(
                            "UPDATE system_tags SET embedding = %s WHERE name = %s",
                            (vec, name)
                        )
                        success_count += 1
                        # 稍微停顿一下，避免触发 API 频率限制
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"处理标签 {name} 失败: {e}")
                
                logger.info(f"向量化任务完成！成功: {success_count} / 总计: {len(tags)}")

if __name__ == "__main__":
    asyncio.run(main())
