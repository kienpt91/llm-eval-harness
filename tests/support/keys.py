"""Secret-shaped strings built at run time, so no literal key-shaped text sits in the repository."""

from __future__ import annotations

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def filler(n: int, seed: int = 7) -> str:
    """Deterministic alphanumeric filler of length ``n``."""
    return "".join(ALNUM[(seed * (i + 3) * 31 + i * i) % len(ALNUM)] for i in range(n))


def aws_key_id() -> str:
    return "AK" + "IA" + "ABCDEFGHIJKLMNOP"


def github_token() -> str:
    return "gh" + "p_" + filler(36)


def payment_key() -> str:
    return "s" + "k_" + "live_" + filler(24)


def sk_style_key() -> str:
    return "s" + "k-" + filler(40)


def slack_token() -> str:
    return "xo" + "xb-" + "1234567890-" + filler(12)


def jwt() -> str:
    return "ey" + "J" + filler(20) + ".ey" + "J" + filler(24) + "." + filler(16)


def private_key_block() -> str:
    return (
        "-" * 5
        + "BEGIN "
        + "RSA PRIVATE "
        + "KEY"
        + "-" * 5
        + "\n"
        + filler(40)
        + "\n"
        + "-" * 5
        + "END "
        + "RSA PRIVATE "
        + "KEY"
        + "-" * 5
    )


def url_with_password() -> str:
    return "post" + "gres://app:" + "hunter2pass" + "@db.internal.example/app"


def quoted_assignment() -> str:
    return "api_" + 'key = "' + filler(32) + '"'


def bearer() -> str:
    return "Bear" + "er " + filler(30)


def hex_key() -> str:
    return "0" + "x" + "ab12cd34" * 8


def all_secrets() -> dict[str, str]:
    return {
        "aws-access-key-id": aws_key_id(),
        "github-token": github_token(),
        "payment-key": payment_key(),
        "sk-style-key": sk_style_key(),
        "slack-token": slack_token(),
        "jwt": jwt(),
        "private-key-block": private_key_block(),
        "url-with-password": url_with_password(),
        "quoted-secret-assignment": quoted_assignment(),
        "bearer-token": bearer(),
        "raw-private-hex": hex_key(),
    }
