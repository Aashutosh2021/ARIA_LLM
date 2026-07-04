"""
AIRA-LLM
Dataset Cleaner
"""

import re


class DatasetCleaner:

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    # ---------------------------------
    # Main Clean Function
    # ---------------------------------
    def clean(self, text: str) -> str:

        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        if self.lowercase:
            text = text.lower()

        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_html(text)
        text = self.remove_extra_spaces(text)
        text = self.remove_extra_newlines(text)

        return text.strip()

    # ---------------------------------
    # Remove URLs
    # ---------------------------------
    def remove_urls(self, text):

        return re.sub(r"http\S+|www\S+", "", text)

    # ---------------------------------
    # Remove Emails
    # ---------------------------------
    def remove_emails(self, text):

        return re.sub(r"\S+@\S+", "", text)

    # ---------------------------------
    # Remove HTML
    # ---------------------------------
    def remove_html(self, text):

        return re.sub(r"<.*?>", "", text)

    # ---------------------------------
    # Remove Multiple Spaces
    # ---------------------------------
    def remove_extra_spaces(self, text):

        return re.sub(r"\s+", " ", text)

    # ---------------------------------
    # Remove Multiple Newlines
    # ---------------------------------
    def remove_extra_newlines(self, text):

        return re.sub(r"\n+", "\n", text)