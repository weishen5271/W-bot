#!/usr/bin/env python3
"""
Email Sender - Send emails via SMTP with flexible configuration.
"""

import argparse
import json
import logging
import os
import smtplib
import sys
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid, parseaddr
from typing import Any, Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EmailConfig:
    """Email configuration manager."""

    DEFAULT_CONFIG = {
        "smtp_server": "",
        "smtp_port": 465,
        "username": "",
        "password": "",
        "from_email": "",
        "security": "ssl",
        "timeout": 30,
        "debug_smtp": False,
    }

    PROVIDERS = {
        "qq": {"smtp_server": "smtp.qq.com", "smtp_port": 465, "security": "ssl"},
        "gmail": {"smtp_server": "smtp.gmail.com", "smtp_port": 587, "security": "tls"},
        "outlook": {"smtp_server": "smtp.office365.com", "smtp_port": 587, "security": "tls"},
        "163": {"smtp_server": "smtp.163.com", "smtp_port": 465, "security": "ssl"},
    }

    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = cls.DEFAULT_CONFIG.copy()
        env_mapping = {
            "EMAIL_SMTP_SERVER": "smtp_server",
            "EMAIL_SMTP_PORT": "smtp_port",
            "EMAIL_USERNAME": "username",
            "EMAIL_PASSWORD": "password",
            "EMAIL_FROM": "from_email",
            "EMAIL_SMTP_SECURITY": "security",
            "EMAIL_TIMEOUT": "timeout",
            "EMAIL_SMTP_DEBUG": "debug_smtp",
        }

        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if not value:
                continue
            if config_key in {"smtp_port", "timeout"}:
                config[config_key] = int(value)
            elif config_key == "debug_smtp":
                config[config_key] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                config[config_key] = value

        if not config["from_email"] and config["username"]:
            config["from_email"] = config["username"]
        return config

    @classmethod
    def from_file(cls, config_path: str = "email_config.json") -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config = cls.DEFAULT_CONFIG.copy()
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.warning("Failed to load config file %s: %s", config_path, e)

        if not config.get("from_email") and config.get("username"):
            config["from_email"] = config["username"]
        return config

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        for field in ["smtp_server", "smtp_port", "username", "password"]:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")

        if config.get("smtp_port") and not isinstance(config["smtp_port"], int):
            errors.append("smtp_port must be an integer")

        if config.get("timeout") and not isinstance(config["timeout"], int):
            errors.append("timeout must be an integer")

        if config.get("security") not in ["ssl", "tls", "none"]:
            errors.append("security must be 'ssl', 'tls', or 'none'")

        for field in ["username", "from_email"]:
            value = config.get(field)
            if value and "@" not in parseaddr(str(value))[1]:
                errors.append(f"{field} must be a valid email address")

        return errors


class EmailSender:
    """Main email sender class."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()

    def validate_config(self) -> None:
        errors = EmailConfig.validate(self.config)
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

    def create_message(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
        attachments: Optional[List[str]] = None,
    ) -> MIMEMultipart:
        """Create email message with optional attachments."""
        msg = MIMEMultipart()
        from_email = self.config.get("from_email") or self.config["username"]
        sender_name = str(Header(from_email, "utf-8"))
        domain = parseaddr(from_email)[1].split("@")[-1] or None
        msg["From"] = formataddr((sender_name, from_email))
        msg["To"] = formataddr(("收件人", to))
        msg["Subject"] = str(Header(subject, "utf-8"))
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain=domain)
        msg["Reply-To"] = from_email
        msg["X-Mailer"] = "W-bot Email Sender"

        body_part = MIMEText(body, "html" if html else "plain", "utf-8")
        msg.attach(body_part)

        for attachment_path in attachments or []:
            if not os.path.exists(attachment_path):
                logger.warning("Attachment not found: %s", attachment_path)
                continue
            try:
                with open(attachment_path, "rb") as f:
                    attachment_data = f.read()
                filename = os.path.basename(attachment_path)
                attachment = MIMEApplication(attachment_data, Name=filename)
                attachment["Content-Disposition"] = f'attachment; filename="{filename}"'
                msg.attach(attachment)
                logger.info("Attached: %s", filename)
            except Exception as e:
                logger.error("Failed to attach %s: %s", attachment_path, e)

        return msg

    def _connect_server(self) -> smtplib.SMTP:
        security = self.config.get("security", "ssl")
        timeout = self.config.get("timeout", 30)
        logger.info(
            "Connecting to %s:%s with %s security...",
            self.config["smtp_server"],
            self.config["smtp_port"],
            security,
        )

        if security == "ssl":
            server = smtplib.SMTP_SSL(
                self.config["smtp_server"],
                self.config["smtp_port"],
                timeout=timeout,
            )
        else:
            server = smtplib.SMTP(
                self.config["smtp_server"],
                self.config["smtp_port"],
                timeout=timeout,
            )
            server.ehlo()
            if security == "tls":
                server.starttls()
                server.ehlo()

        if self.config.get("debug_smtp"):
            server.set_debuglevel(1)
        return server

    def send(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
        attachments: Optional[List[str]] = None,
    ) -> bool:
        """Send email via SMTP.

        A True result means the upstream SMTP server accepted the request.
        It does not guarantee inbox delivery.
        """
        server: smtplib.SMTP | None = None
        try:
            msg = self.create_message(to, subject, body, html, attachments)
            server = self._connect_server()
            logger.info("Logging in as %s...", self.config["username"])
            server.login(self.config["username"], self.config["password"])

            logger.info("Sending email to %s...", to)
            from_email = self.config.get("from_email") or self.config["username"]
            result = server.sendmail(from_email, [to], msg.as_string())

            if result:
                logger.error("SMTP accepted the request but refused recipients: %s", result)
                return False

            logger.info("SMTP accepted the email request successfully.")
            logger.info("Acceptance by SMTP does not guarantee inbox delivery.")
            return True
        except smtplib.SMTPAuthenticationError:
            logger.error("Authentication failed. Check username and password/authorization code.")
            return False
        except smtplib.SMTPRecipientsRefused as e:
            logger.error("Recipient refused by SMTP server: %s", e.recipients)
            return False
        except smtplib.SMTPDataError as e:
            logger.error("SMTP data error: code=%s, message=%s", e.smtp_code, e.smtp_error)
            return False
        except smtplib.SMTPConnectError:
            logger.error("Connection failed to %s:%s", self.config["smtp_server"], self.config["smtp_port"])
            return False
        except Exception as e:
            logger.error("Failed to send email: %s", e)
            return False
        finally:
            if server is not None:
                try:
                    server.quit()
                except Exception:
                    pass

    @classmethod
    def create_from_env(cls):
        return cls(EmailConfig.from_env())

    @classmethod
    def create_from_file(cls, config_path: str = "email_config.json"):
        return cls(EmailConfig.from_file(config_path))


def send_email_cli() -> None:
    """Command line interface for sending emails."""
    parser = argparse.ArgumentParser(description="Send email via SMTP")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    send_parser = subparsers.add_parser("send", help="Send an email")
    send_parser.add_argument("--to", required=True, help="Recipient email address")
    send_parser.add_argument("--subject", required=True, help="Email subject")
    send_parser.add_argument("--body", required=True, help="Email body content")
    send_parser.add_argument("--html", action="store_true", help="Body is HTML content")
    send_parser.add_argument("--attach", action="append", help="Attachment file path")
    send_parser.add_argument("--smtp-server", help="SMTP server address")
    send_parser.add_argument("--smtp-port", type=int, help="SMTP port")
    send_parser.add_argument("--username", help="SMTP username/email")
    send_parser.add_argument("--password", help="SMTP password/authorization code")
    send_parser.add_argument("--from", dest="from_email", help="From email address")
    send_parser.add_argument("--security", choices=["ssl", "tls", "none"], help="Security type")
    send_parser.add_argument("--config", help="Configuration file path")
    send_parser.add_argument(
        "--debug-smtp",
        action="store_true",
        help="Enable raw SMTP protocol logs for troubleshooting",
    )

    config_parser = subparsers.add_parser("config", help="Show current configuration")
    config_parser.add_argument("--provider", choices=["qq", "gmail", "outlook", "163"], help="Show provider configuration")

    args = parser.parse_args()

    if args.command == "send":
        if args.config:
            config = EmailConfig.from_file(args.config)
        else:
            config = EmailConfig.from_env()

        if args.smtp_server:
            config["smtp_server"] = args.smtp_server
        if args.smtp_port:
            config["smtp_port"] = args.smtp_port
        if args.username:
            config["username"] = args.username
        if args.password:
            config["password"] = args.password
        if args.from_email:
            config["from_email"] = args.from_email
        if args.security:
            config["security"] = args.security
        if args.debug_smtp:
            config["debug_smtp"] = True

        sender = EmailSender(config)
        try:
            sender.validate_config()
        except ValueError as e:
            logger.error(str(e))
            logger.error("Please set configuration via:")
            logger.error("  - Environment variables (EMAIL_SMTP_SERVER, EMAIL_USERNAME, etc.)")
            logger.error("  - Command line arguments (--smtp-server, --username, etc.)")
            logger.error("  - Configuration file (--config path/to/config.json)")
            sys.exit(1)

        success = sender.send(
            to=args.to,
            subject=args.subject,
            body=args.body,
            html=args.html,
            attachments=args.attach,
        )
        sys.exit(0 if success else 1)

    if args.command == "config":
        if args.provider:
            provider_config = EmailConfig.PROVIDERS.get(args.provider, {})
            print(f"{args.provider.capitalize()} configuration:")
            for key, value in provider_config.items():
                print(f"  {key}: {value}")
            return

        env_config = EmailConfig.from_env()
        print("Current configuration from environment variables:")
        for key, value in env_config.items():
            if key == "password" and value:
                print(f"  {key}: [SET]")
            elif value != "" and value is not False:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: [NOT SET]")
        return

    parser.print_help()


def test_connection() -> bool:
    """Test SMTP connection with current configuration."""
    server: smtplib.SMTP | None = None
    try:
        sender = EmailSender.create_from_env()
        sender.validate_config()
        print("Configuration is valid!")
        print(f"SMTP Server: {sender.config['smtp_server']}:{sender.config['smtp_port']}")
        print(f"Username: {sender.config['username']}")
        print(f"Security: {sender.config.get('security', 'ssl')}")
        print(f"SMTP Debug: {bool(sender.config.get('debug_smtp'))}")

        print("\nTesting connection...")
        server = sender._connect_server()
        print("Connection successful!")

        print("Testing authentication...")
        server.login(sender.config["username"], sender.config["password"])
        print("Authentication successful!")
        print("SMTP connection is healthy, but inbox delivery still depends on the recipient server.")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass


if __name__ == "__main__":
    send_email_cli()
