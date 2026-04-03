# Email Sender Skill

从配置文件读取用户配置，发送邮件。

## 文件说明

- `config.json` - 配置文件（发件人、收件人、授权码等）
- `email_sender.py` - 核心邮件发送模块
- `send_email.py` - 发送脚本

## 使用方法

```bash
# 发送邮件（使用配置文件中的默认收件人）
python send_email.py

# 指定收件人
python send_email.py --to "test@example.com"

# 指定主题和内容
python send_email.py --to "test@example.com" --subject "标题" --body "内容"
```

## 配置文件 (config.json)

```json
{
  "smtp_server": "smtp.qq.com",
  "smtp_port": 465,
  "username": "your_email@qq.com",
  "password": "your_authorization_code",
  "from_email": "your_email@qq.com",
  "default_recipient": "recipient@example.com",
  "security": "ssl",
  "timeout": 30
}
```

## Python代码调用

```python
from send_email import send_email

# 发送邮件（使用默认配置）
success = send_email(
    subject="邮件主题",
    body="邮件内容"
)

# 指定收件人
success = send_email(
    to="recipient@example.com",
    subject="邮件主题",
    body="邮件内容"
)
```