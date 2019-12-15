from celery import Celery

celery = Celery(broker='redis://localhost:6379/0')