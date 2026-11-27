"""Regresión robusta Huber para RNA, CNA y RNA+CNA."""
from regression_shared_34 import train_model

if __name__ == "__main__":
    train_model("huber")
