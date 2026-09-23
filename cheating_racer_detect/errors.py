class AppError(Exception):
    """A stable public error, without raw media paths or tool output."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
