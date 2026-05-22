from importlib.metadata import PackageNotFoundError, version


def main():
    """Main entry point for the package."""
    from src.lindorm_mcp_server import server

    server.main()

__all__ = ['main']
try:
    __version__ = version("alibabacloud-lindorm-mcp-server")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
