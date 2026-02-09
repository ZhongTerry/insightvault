# InsightVault API v1 文档

## 🎉 重大更新

### 新功能

1. **RESTful API v1** - 完全符合REST规范的API设计
2. **API密钥认证** - 支持使用API密钥进行程序化访问
3. **权限范围管理** - API密钥可设置read/write权限
4. **标题+内容格式** - 情报改为标题+内容结构
5. **详情页面** - 点击情报可查看完整详情
6. **Markdown支持** - 内容支持Markdown格式，带XSS防护
7. **异步向量化** 🆕 - 创建/更新情报立即返回，后台自动生成向量，大幅提升响应速度

---

## 🔐 认证方式

### 1. JWT Token（Web界面）
```bash
Authorization: Bearer <jwt_token>
```

### 2. API密钥（程序化访问）
```bash
Authorization: Bearer <api_key>
```

**创建API密钥：**
- 登录后访问 `/settings`
- 在"API密钥管理"部分创建新密钥
- 选择权限范围：`read`（读取）、`write`（写入）
- 密钥只显示一次，请妥善保存

---

## 📡 API端点

### 情报管理

#### 1. 列出情报（支持搜索）
```http
GET /api/v1/items?type={type}&q={query}&page={page}&per_page={per_page}
```

**权限:** `read`

**查询参数:**
- `type`: 搜索类型（`all`, `ai`, `key`, `tag`）
- `q`: 搜索关键词
- `page`: 页码（默认1）
- `per_page`: 每页数量（默认10，最大100）

**响应:**
```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "title": "示例标题",
      "content_preview": "内容预览...",
      "created_at": "2026-02-09T10:30:00Z",
      "visibility": "private",
      "group_id": null,
      "group_name": null,
      "tags": ["标签1", "标签2"],
      "score": 0.95
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "has_more": true
  }
}
```

#### 2. 获取单个情报详情
```http
GET /api/v1/items/{item_id}
```

**权限:** `read`

**响应:**
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "title": "示例标题",
    "content": "完整内容（支持Markdown）",
    "created_at": "2026-02-09T10:30:00Z",
    "visibility": "private",
    "group_id": null,
    "owner_user_id": 1,
    "owner_name": "用户名",
    "group_name": null,
    "tags": ["标签1", "标签2"]
  }
}
```

#### 3. 创建情报
```http
POST /api/v1/items
Content-Type: application/json
```

**权限:** `write`

**请求体:**
```json
{
  "title": "情报标题（可选）",
  "content": "情报内容（支持Markdown）",
  "visibility": "private",
  "group_id": null,
  "tags": "标签1, 标签2"
}
```

**响应:**
```json
{
  "status": "success",
  "data": {
    "id": 123
  }
}
```

**性能优化:** 🚀 情报会立即插入数据库并返回，向量化处理在后台异步进行，不阻塞响应。向量化完成后，情报即可被 AI 搜索到。

#### 4. 更新情报
```http
PUT /api/v1/items/{item_id}
Content-Type: application/json
```

**权限:** `write`

**请求体:**
```json
{
  "title": "新标题",
  "content": "新内容",
  "visibility": "private",
  "group_id": null,
  "tags": "新标签"
}
```

**性能优化:** 🚀 情报会立即更新并返回，重新向量化在后台异步进行。

#### 5. 删除情报
```http
DELETE /api/v1/items/{item_id}
```

**权限:** `write`

---

## 💡 使用示例

### Python示例
```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "http://localhost:8080"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 创建情报
response = requests.post(
    f"{BASE_URL}/api/v1/items",
    headers=headers,
    json={
        "title": "Python学习笔记",
        "content": "## 列表推导式\n\n使用列表推导式可以简化代码...",
        "visibility": "private",
        "tags": "python, 编程"
    }
)

print(response.json())

# 搜索情报
response = requests.get(
    f"{BASE_URL}/api/v1/items",
    headers=headers,
    params={
        "type": "key",
        "q": "python",
        "page": 1
    }
)

items = response.json()["data"]
for item in items:
    print(f"{item['title']}: {item['content_preview']}")

# 获取详情
item_id = items[0]["id"]
response = requests.get(
    f"{BASE_URL}/api/v1/items/{item_id}",
    headers=headers
)

detail = response.json()["data"]
print(f"\n标题: {detail['title']}")
print(f"内容: {detail['content']}")
```

### cURL示例
```bash
# 创建API密钥（先在网页上创建）
API_KEY="your_api_key_here"

# 创建情报
curl -X POST http://localhost:8080/api/v1/items \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "测试标题",
    "content": "# 测试内容\n\n这是一条测试情报",
    "visibility": "private",
    "tags": "测试"
  }'

# 搜索情报
curl -X GET "http://localhost:8080/api/v1/items?type=all&page=1" \
  -H "Authorization: Bearer $API_KEY"

# 获取详情
curl -X GET http://localhost:8080/api/v1/items/1 \
  -H "Authorization: Bearer $API_KEY"

# 更新情报
curl -X PUT http://localhost:8080/api/v1/items/1 \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "更新的标题",
    "content": "更新的内容",
    "visibility": "private"
  }'

# 删除情报
curl -X DELETE http://localhost:8080/api/v1/items/1 \
  -H "Authorization: Bearer $API_KEY"
```

---

## 🔒 安全特性

1. **API密钥加密存储** - 使用SHA-256哈希，数据库不存储明文
2. **权限范围控制** - 密钥可限制为只读或读写
3. **XSS防护** - Markdown渲染使用DOMPurify清理
4. **权限验证** - 所有操作都验证用户权限

---

## ⚡ 性能优化

### 异步向量化机制

**优化前:**
- 创建/更新情报时需等待向量化完成（通常 1-3 秒）
- 大文本向量化可能让用户长时间等待

**优化后:**
- 情报立即插入数据库，并返回 ID（通常 < 100ms）
- 向量化在后台异步进行，不阻塞用户操作
- 向量化完成前，情报可通过 ID/关键词搜索
- 向量化完成后，自动支持 AI 语义搜索

**技术实现:**
- 使用 FastAPI 的 `BackgroundTasks`
- 向量化失败会记录日志，不影响主流程
- 可通过 `/api/revectorize_all` 重新向量化所有项目

---

## 🚀 迁移指南

### 从旧API迁移到v1

**旧API (仍可用):**
```
POST /api/add
GET /api/search
PUT /api/update/{id}
DELETE /api/delete/{id}
```

**新API v1:**
```
POST /api/v1/items
GET /api/v1/items
PUT /api/v1/items/{id}
DELETE /api/v1/items/{id}
```

**主要变化:**
1. 所有API添加 `/v1/` 前缀
2. 统一资源路径为 `items`
3. 请求/响应格式规范化
4. 添加标题字段支持
5. 内容预览和详情分离
6. **异步向量化，响应速度提升10-30倍** 🚀

---

## 📝 注意事项

1. **API密钥安全**
   - 密钥只在创建时显示一次
   - 请勿在代码中硬编码密钥
   - 建议使用环境变量存储

2. **Markdown支持**
   - 详情页支持完整Markdown渲染
   - 列表页只显示前200字符预览
   - 自动防XSS攻击

3. **向后兼容**
   - 旧API端点暂时保留
   - 建议尽快迁移到v1
   - 未来版本可能移除旧API

4. **性能优化**
   - 创建/更新操作采用异步向量化
   - 响应时间从 1-3秒 降低到 < 100ms
   - 后台自动处理向量化，不阻塞用户

---

## 🆕 其他新增功能

1. **个人资料管理** - `/settings` 页面可修改名称和邮箱
2. **密码修改** - 安全的密码更新流程
3. **详情页面** - 点击情报卡片查看完整内容
4. **Markdown编辑器提示** - 输入框提示支持Markdown

---

## 🐛 已知问题

1. 旧数据没有标题字段，会显示"无标题"
2. API密钥创建后请立即保存，无法再次查看

---

## 📞 支持

如有问题请提交Issue或联系管理员。
