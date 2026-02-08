import feedparser
from datetime import datetime

# 定义源，方便统一管理
RSS_FEEDS = {
    "📰 人民日报微信版": "http://feedmaker.kindle4rss.com/feeds/rmrbwx.weixin.xml"
}

def get_rss_data(limit=10):
    """
    获取 RSS 数据并返回结构化列表
    :param limit: 每个源获取的条数
    :return: 包含情报字典的列表
    """
    all_results = []

    for name, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        
        # 提取前 limit 条有效条目
        entries = feed.entries[:limit]
        
        for entry in entries:
            item = {
                "source": name,
                "title": entry.get('title', '无标题'),
                "link": entry.get('link', '无链接'),
                "published": entry.get('published', '未知时间'),
                "summary": entry.get('summary', ''), # 部分源有摘要
                "fetched_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            all_results.append(item)
            
    return all_results

def display_rss_data(news_list):
    """
    负责将获取到的结构化数据打印到控制台
    """
    if not news_list:
        print("未获取到任何数据。")
        return

    print(f"=== InsightVault 情报抓取报告 ({news_list[0]['fetched_at']}) ===\n")
    
    current_source = ""
    for i, item in enumerate(news_list, 1):
        # 当来源变化时，打印一个明显的分割线
        if item['source'] != current_source:
            current_source = item['source']
            print(f"\n【{current_source}】" + "="*40)
        
        print(f"{i}. {item['title']}")
        print(f"   时间: {item['published']}")
        print(f"   链接: {item['link']}")
        print("-" * 30)

# --- 下面是测试逻辑，只有直接运行本脚本时才会执行 ---
if __name__ == "__main__":
    # 1. 获取数据（这一步可以被其他文件调用）
    data = get_rss_data(limit=10)
    
    # 2. 展示数据
    display_rss_data(data)