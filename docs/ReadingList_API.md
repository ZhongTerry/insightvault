# InsightVault 稍后阅读 API 文档

## 概述

稍后阅读功能允许用户通过浏览器插件或其他工具快速保存网页内容，用于后续阅读和整理。本文档面向浏览器插件开发者，说明如何与 InsightVault 的稍后阅读 API 进行交互。

## 功能特性

### 两种保存模式

1. **完整内容模式**（推荐）
   - 插件成功解析网页的文字内容
   - 保存：标题 + URL + 解析的文字内容
   - 用户体验：可以在 InsightVault 内直接阅读文章，无需打开原网页

2. **仅链接模式**（备用）
   - 插件无法解析网页内容（如反爬虫保护、动态加载等）
   - 保存：标题 + URL
   - 用户体验：点击后在新标签页打开原网页

### 核心功能

- ✅ 快速保存网页到阅读列表
- ✅ 标记已读/未读状态
- ✅ 归档管理
- ✅ 将有价值的内容保存到情报库（自动向量化，支持 AI 搜索）
- ✅ 自动提取来源信息

---

## 认证方式

所有 API 请求都需要使用 Bearer Token 认证。插件用户需要在 InsightVault 设置中生成 API 密钥。

```http
Authorization: Bearer YOUR_API_KEY
```

**获取 API 密钥的方式**：
1. 登录 InsightVault
2. 进入"设置"页面
3. 在"API 密钥管理"区域创建新密钥
4. 选择 `read` 和 `write` 权限
5. 复制生成的密钥（仅显示一次）

---

## API 端点

### 1. 添加阅读项

**端点**: `POST /api/v1/reading-list`

#### 请求格式

```json
{
  "title": "文章标题（必填）",
  "url": "https://example.com/article（必填）",
  "content": "解析的文章文字内容（可选，支持 Markdown）",
  "source": "来源网站名称（可选）",
  "cover_image": "封面图 URL（可选）"
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | string | ✅ | 文章标题，显示在列表中 |
| `url` | string | ✅ | 文章原始 URL，用于跳转 |
| `content` | string | ❌ | 解析的文章内容，支持纯文本或 Markdown。如果提供，用户可以直接阅读；如果为空，点击时会打开原网页 |
| `source` | string | ❌ | 来源网站名称（如"人民日报"），显示在文章元信息中。如果不提供，系统会从 URL 提取域名 |
| `cover_image` | string | ❌ | 文章封面图 URL（当前版本未在界面显示，为未来功能预留） |

#### 响应示例

```json
{
  "status": "success",
  "data": {
    "id": 42
  }
}
```

#### 插件实现建议

```javascript
// 浏览器插件伪代码示例
async function saveToInsightVault(apiKey) {
  const articleData = {
    title: document.title,
    url: window.location.href,
    source: extractSourceName(), // 例如从 meta 标签获取
    content: null,
    cover_image: null
  };

  // 尝试解析文章内容
  try {
    articleData.content = extractArticleContent(); // 使用 Readability.js 或类似库
    articleData.cover_image = extractCoverImage(); // 提取 og:image 等
  } catch (error) {
    console.warn('无法解析内容，将以仅链接模式保存');
  }

  const response = await fetch('https://your-insightvault.com/api/v1/reading-list', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(articleData)
  });

  return await response.json();
}
```

---

### 2. 获取阅读列表

**端点**: `GET /api/v1/reading-list`

#### 查询参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filter` | string | `all` | 筛选条件：`all`（全部）、`unread`（未读）、`read`（已读）、`archived`（已归档） |
| `page` | int | 1 | 页码（从 1 开始） |
| `per_page` | int | 20 | 每页条数（最大 100） |

#### 响应示例

```json
{
  "status": "success",
  "data": [
    {
      "id": 42,
      "title": "文章标题",
      "url": "https://example.com/article",
      "source": "Example Site",
      "cover_image": null,
      "has_content": true,
      "is_read": false,
      "is_archived": false,
      "created_at": "2026-02-09T10:30:00+08:00"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "has_more": false
  }
}
```

---

### 3. 获取单篇文章详情

**端点**: `GET /api/v1/reading-list/{item_id}`

#### 响应示例

```json
{
  "status": "success",
  "data": {
    "id": 42,
    "title": "文章标题",
    "url": "https://example.com/article",
    "content": "文章完整内容...",
    "source": "Example Site",
    "cover_image": null,
    "has_content": true,
    "owner_user_id": 1,
    "is_read": false,
    "is_archived": false,
    "created_at": "2026-02-09T10:30:00+08:00",
    "updated_at": "2026-02-09T10:30:00+08:00"
  }
}
```

---

### 4. 更新阅读项

**端点**: `PUT /api/v1/reading-list/{item_id}`

#### 请求格式

```json
{
  "is_read": true,      // 可选：标记为已读/未读
  "is_archived": false  // 可选：归档/取消归档
}
```

#### 使用场景

- 标记已读：用户阅读完文章后
- 归档：用户想隐藏该文章但不删除

---

### 5. 删除阅读项

**端点**: `DELETE /api/v1/reading-list/{item_id}`

#### 响应示例

```json
{
  "status": "success"
}
```

---

### 6. 保存到情报库

**端点**: `POST /api/v1/reading-list/{item_id}/save-to-vault`

#### 功能说明

将阅读列表中有完整内容的文章保存到 InsightVault 情报库，自动进行向量化处理，使其可以被 AI 语义搜索。

#### 限制条件

- 仅支持 `has_content = true` 的文章
- 保存后自动归档该阅读项（`is_archived = true`）

#### 响应示例

```json
{
  "status": "success",
  "data": {
    "vault_id": 123
  }
}
```

---

## 浏览器插件最佳实践

### 1. 内容解析建议

使用成熟的内容提取库：

- **Readability.js**（Mozilla 开发）：擅长提取文章主体
- **Turndown**：将 HTML 转换为 Markdown
- **Mercury Parser**：强大的内容提取工具

示例：
```javascript
import Readability from '@mozilla/readability';
import TurndownService from 'turndown';

function extractArticleContent() {
  const documentClone = document.cloneNode(true);
  const reader = new Readability(documentClone);
  const article = reader.parse();
  
  if (!article) return null;
  
  // 转换为 Markdown
  const turndownService = new TurndownService();
  return turndownService.turndown(article.content);
}
```

### 2. 来源识别

优先级顺序：
1. `<meta property="og:site_name" content="网站名">`
2. `<meta name="author" content="作者">`
3. 从 URL 提取域名

```javascript
function extractSourceName() {
  const ogSiteName = document.querySelector('meta[property="og:site_name"]');
  if (ogSiteName) return ogSiteName.content;
  
  const author = document.querySelector('meta[name="author"]');
  if (author) return author.content;
  
  return new URL(window.location.href).hostname;
}
```

### 3. 错误处理

```javascript
async function saveToInsightVault(apiKey, articleData) {
  try {
    const response = await fetch('https://your-insightvault.com/api/v1/reading-list', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(articleData)
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`保存失败: ${error}`);
    }

    return await response.json();
  } catch (error) {
    console.error('InsightVault 保存失败:', error);
    throw error;
  }
}
```

### 4. 用户体验优化

- **即时反馈**：保存成功后显示通知
- **快捷键支持**：例如 `Ctrl+Shift+S` 快速保存
- **右键菜单**：在选中文本时提供"保存到 InsightVault"选项
- **批量保存**：支持一次保存多个标签页

---

## 常见问题

### Q1: 什么情况下会保存为"仅链接"模式？

当 `content` 字段为空或未提供时，InsightVault 会将 `has_content` 设为 `false`，此时：
- 用户点击文章会在新标签页打开原 URL
- 无法保存到情报库（因为没有可向量化的内容）

### Q2: 如何处理付费墙或需要登录的内容？

插件只能保存当前浏览器能访问的内容。如果用户已登录，Readability.js 通常能正确提取。如果内容受限，建议保存为"仅链接"模式。

### Q3: 支持哪些 Markdown 语法？

InsightVault 使用 `marked.js` 渲染 Markdown，支持：
- 标题 (`# ## ###`)
- 列表（有序/无序）
- 代码块
- 引用块
- 粗体/斜体
- 链接和图片
- 表格

### Q4: 如何验证 API 密钥是否有效？

可以调用 `/api/auth/me` 端点：

```javascript
const response = await fetch('https://your-insightvault.com/api/auth/me', {
  headers: { 'Authorization': `Bearer ${apiKey}` }
});

if (response.status === 401) {
  alert('API 密钥无效或已过期');
}
```

---

## 示例：完整的浏览器插件工作流

```javascript
// background.js 或 content.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'saveToInsightVault') {
    handleSave();
  }
});

async function handleSave() {
  const apiKey = await getStoredApiKey(); // 从浏览器存储获取
  
  if (!apiKey) {
    showNotification('请先在插件设置中配置 API 密钥');
    return;
  }

  const articleData = {
    title: document.title,
    url: window.location.href,
    source: extractSourceName(),
    content: null,
    cover_image: null
  };

  // 尝试提取内容
  try {
    const reader = new Readability(document.cloneNode(true));
    const article = reader.parse();
    
    if (article && article.content) {
      const turndown = new TurndownService();
      articleData.content = turndown.turndown(article.content);
    }
    
    const ogImage = document.querySelector('meta[property="og:image"]');
    if (ogImage) {
      articleData.cover_image = ogImage.content;
    }
  } catch (error) {
    console.warn('内容提取失败，将以仅链接模式保存', error);
  }

  // 发送到 InsightVault
  try {
    const response = await fetch('https://your-insightvault.com/api/v1/reading-list', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(articleData)
    });

    if (response.ok) {
      const result = await response.json();
      showNotification('✅ 已保存到 InsightVault 稍后阅读');
    } else {
      throw new Error(await response.text());
    }
  } catch (error) {
    showNotification('❌ 保存失败: ' + error.message);
  }
}
```

---

## 技术架构

### 数据库表结构

```sql
CREATE TABLE reading_list (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content TEXT,                      -- 可为 NULL
    source VARCHAR(255),               -- 来源名称
    cover_image TEXT,                  -- 封面图 URL
    has_content BOOLEAN DEFAULT FALSE, -- 是否有完整内容
    owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_read BOOLEAN DEFAULT FALSE,     -- 是否已读
    is_archived BOOLEAN DEFAULT FALSE, -- 是否归档
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 索引

- `idx_rl_owner`: 按用户查询
- `idx_rl_created`: 按时间排序
- `idx_rl_status`: 按状态筛选（已读/未读/归档）

---

## 版本历史

- **v1.0.0** (2026-02-09): 初始版本，支持基础稍后阅读功能

---

## 联系与支持

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目仓库地址]
- 邮件: [support@example.com]

---

**祝开发顺利！🚀**
