import requests
from bs4 import BeautifulSoup
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from dotenv import load_dotenv

# 加载配置
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(root_dir, ".env")
load_dotenv(env_path)

def get_netease_news_summary():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    final_report = []

    # --- 第一部分：抓取国际要闻 (World) ---
    world_url = "https://news.163.com/world/"
    try:
        resp_world = requests.get(world_url, headers=headers, timeout=10)
        resp_world.encoding = resp_world.apparent_encoding
        soup_world = BeautifulSoup(resp_world.text, 'html.parser')
        
        world_box = soup_world.find('div', class_='hidden')
        if world_box:
            links = world_box.find_all('a')
            for link in links[:10]: # 抓取 10 条
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
            for item in items[:10]: # 抓取 10 条
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

def send_email_report(report):
    if not report:
        print("无内容可发送")
        return

    # 从 .env 读取配置
    smtp_srv = os.getenv("SMTP_SRV")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_ssl = os.getenv("SMTP_SSL", "True").lower() == "true"
    report_email = os.getenv("REPORT_EMAIL")

    if not all([smtp_srv, smtp_user, smtp_pass, report_email]):
        print("错误：SMTP 配置不完整（请检查 .env 中的 SMTP_SRV, SMTP_USER, SMTP_PASS, REPORT_EMAIL）")
        return

    # 构建邮件内容
    content_lines = ["<html><body>", "<h2>网易新闻每日情报简报</h2>", "<hr>"]
    for idx, news in enumerate(report, 1):
        content_lines.append(f"<p><b>{idx:02d}. [{news['cate']}]</b> {news['title']}<br>")
        content_lines.append(f"链接: <a href='{news['url']}'>{news['url']}</a></p>")
        if idx == 10:
            content_lines.append("<hr>")
    content_lines.append("</body></html>")
    
    html_content = "".join(content_lines)

    message = MIMEText(html_content, 'html', 'utf-8')
    message['From'] = smtp_user
    message['To'] = report_email
    message['Subject'] = Header("InsightVault | 网易新闻每日情报报告", 'utf-8')

    try:
        if smtp_ssl:
            server = smtplib.SMTP_SSL(smtp_srv, smtp_port)
        else:
            server = smtplib.SMTP(smtp_srv, smtp_port)
            server.starttls()
            
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [report_email], message.as_string())
        server.quit()
        print(f"邮件发送成功！收件人: {report_email}")
    except Exception as e:
        print(f"邮件发送失败: {e}")

if __name__ == "__main__":
    news = get_netease_news_summary()
    if news:
        send_email_report(news)
    else:
        print("采集失败，未发送邮件。")
