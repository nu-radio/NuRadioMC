import os

if "DATABASE_TARGET" in os.environ:
    DATABASE_TARGET = os.environ.get("DATABASE_TARGET")
else:
    DATABASE_TARGET = "env_pw_user"
