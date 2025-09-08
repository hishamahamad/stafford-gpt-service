import time
import psycopg2

from ..config.settings import settings


def connect_to_database(max_retries: int = 5, delay: int = 5) -> psycopg2.extensions.connection:
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(settings.database_url)
            return conn
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise  # Let the caller handle the exception
        except Exception as e:
            raise  # Let the caller handle the exception


def get_database_connection() -> psycopg2.extensions.connection:
    return connect_to_database()
