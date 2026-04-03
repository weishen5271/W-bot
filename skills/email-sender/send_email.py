#!/usr/bin/env python3
"""
简单的邮件发送脚本 - 使用配置文件
"""

import json
import sys
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from email_sender import EmailSender


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please make sure config.json exists")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Failed to load config: {e}")
        sys.exit(1)


def send_email(to=None, subject="Test Email", body="This is a test email."):
    """
    Send email

    Args:
        to: Recipient email, uses config default if None
        subject: Email subject
        body: Email body
    """
    # Load config
    config = load_config()

    # Get recipient
    recipient = to or config.get("default_recipient")
    if not recipient:
        print("Error: No recipient specified")
        print("Set default_recipient in config.json or specify --to")
        return False

    # Extract email settings from config
    email_config = {
        'smtp_server': config['smtp_server'],
        'smtp_port': config['smtp_port'],
        'username': config['username'],
        'password': config['password'],
        'from_email': config['from_email'],
        'security': config.get('security', 'ssl'),
        'timeout': config.get('timeout', 30)
    }

    print("Sending email...")
    print(f"  From: {email_config['from_email']}")
    print(f"  To: {recipient}")
    print(f"  Subject: {subject}")

    try:
        # Create sender
        sender = EmailSender(email_config)

        # Send email
        success = sender.send(
            to=recipient,
            subject=subject,
            body=body
        )

        if success:
            print("[OK] Email sent successfully!")
            return True
        else:
            print("[FAIL] Email send failed")
            return False

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send email")
    parser.add_argument("--to", help="Recipient email address")
    parser.add_argument("--subject", default="Test Email", help="Email subject")
    parser.add_argument("--body", default="This is a test email.", help="Email body")

    args = parser.parse_args()

    success = send_email(
        to=args.to,
        subject=args.subject,
        body=args.body
    )

    sys.exit(0 if success else 1)