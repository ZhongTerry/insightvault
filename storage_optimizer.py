"""
存储优化模块：版本控制与溯源系统
支持数据溯源、版本历史、变更追踪
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("InsightVault.Storage")


@dataclass
class ProvenanceRecord:
    """数据溯源记录"""
    source_type: str  # url, upload, api, import
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    original_filename: Optional[str] = None
    importer: Optional[str] = None  # 导入者
    import_time: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VersionChange:
    """版本变更记录"""
    version: int
    changed_by: int  # user_id
    changed_at: str
    change_type: str  # create, update, merge, split
    diff_summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProvenanceManager:
    """溯源管理器"""
    
    @staticmethod
    def create_provenance(
        source_type: str,
        source_url: Optional[str] = None,
        source_file: Optional[str] = None,
        original_filename: Optional[str] = None,
        importer: Optional[str] = None,
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """创建溯源记录"""
        record = ProvenanceRecord(
            source_type=source_type,
            source_url=source_url,
            source_file=source_file,
            original_filename=original_filename,
            importer=importer,
            import_time=datetime.utcnow().isoformat(),
            checksum=checksum,
            metadata=metadata or {}
        )
        return asdict(record)
    
    @staticmethod
    def create_version_change(
        version: int,
        changed_by: int,
        change_type: str,
        diff_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """创建版本变更记录"""
        change = VersionChange(
            version=version,
            changed_by=changed_by,
            changed_at=datetime.utcnow().isoformat(),
            change_type=change_type,
            diff_summary=diff_summary,
            metadata=metadata or {}
        )
        return asdict(change)


class StorageOptimizer:
    """存储优化器"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def add_metadata_column_if_not_exists(self):
        """自动为表添加元数据列"""
        async with self.db_pool.connection() as conn:
            # 为 cloud_items 添加元数据字段
            await conn.execute("""
                ALTER TABLE cloud_items 
                ADD COLUMN IF NOT EXISTS extracted_metadata JSONB,
                ADD COLUMN IF NOT EXISTS extracted_content TEXT,
                ADD COLUMN IF NOT EXISTS provenance JSONB,
                ADD COLUMN IF NOT EXISTS version_history JSONB;
            """)
            
            # 为 intelligence_vault 添加溯源字段
            await conn.execute("""
                ALTER TABLE intelligence_vault
                ADD COLUMN IF NOT EXISTS provenance JSONB,
                ADD COLUMN IF NOT EXISTS version_history JSONB,
                ADD COLUMN IF NOT EXISTS source_file_id INTEGER REFERENCES cloud_items(id);
            """)
            
            # 创建索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cloud_metadata ON cloud_items USING gin(extracted_metadata);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vault_provenance ON intelligence_vault USING gin(provenance);
            """)
            
            logger.info("✓ 存储优化：元数据列已确保存在")
    
    async def store_file_metadata(
        self,
        item_id: int,
        extracted_content: str,
        metadata: Dict[str, Any],
        provenance: Dict[str, Any]
    ):
        """存储文件解析元数据"""
        async with self.db_pool.connection() as conn:
            await conn.execute(
                """
                UPDATE cloud_items
                SET 
                    extracted_metadata = %s,
                    extracted_content = %s,
                    provenance = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (json.dumps(metadata), extracted_content, json.dumps(provenance), item_id)
            )
            logger.info(f"✓ 已保存文件 {item_id} 的元数据和提取内容")
    
    async def store_version_history(self, item_id: int, version_change: Dict):
        """添加版本历史记录"""
        async with self.db_pool.connection() as conn:
            # 读取现有历史
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT version_history FROM cloud_items WHERE id = %s",
                    (item_id,)
                )
                row = await cur.fetchone()
                history = json.loads(row[0]) if row and row[0] else []
            
            # 追加新记录
            history.append(version_change)
            
            # 更新数据库
            await conn.execute(
                "UPDATE cloud_items SET version_history = %s WHERE id = %s",
                (json.dumps(history), item_id)
            )
            logger.info(f"✓ 已添加版本历史记录到文件 {item_id}")
    
    async def get_item_provenance(self, item_id: int) -> Optional[Dict]:
        """获取数据溯源信息"""
        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT provenance FROM cloud_items WHERE id = %s",
                    (item_id,)
                )
                row = await cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        return None
    
    async def get_version_history(self, item_id: int) -> List[Dict]:
        """获取版本历史"""
        async with self.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT version_history FROM cloud_items WHERE id = %s",
                    (item_id,)
                )
                row = await cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        return []


# 工厂函数
def create_storage_optimizer(db_pool) -> StorageOptimizer:
    """创建存储优化器实例"""
    return StorageOptimizer(db_pool)
