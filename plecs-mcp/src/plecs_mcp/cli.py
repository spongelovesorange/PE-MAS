from __future__ import annotations

from .server import create_server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
