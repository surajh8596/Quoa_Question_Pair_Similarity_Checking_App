import re
import contractions

def clean_test_for_sample_data(text):
    text=str(text).lower()
    text=text.replace('%', ' percent')
    text=text.replace('$', ' dollar ')
    text=text.replace('₹', ' rupee ')
    text=text.replace('€', ' euro ')
    text=text.replace('@', ' at ')
    text=text.replace(',000,000,000 ', 'b ')
    text=text.replace(',000,000 ', 'm ')
    text=text.replace(',000 ', 'k ')
    text=re.sub(r'([0-9]+)000000000', r'\1b', text)
    text=re.sub(r'([0-9]+)000000', r'\1m', text)
    text=re.sub(r'([0-9]+)000', r'\1k', text)
    pattern=re.compile('\W')
    text=re.sub(pattern, ' ', text).strip()
    text=re.sub("<.*?>", "", text)
    text=contractions.fix(text)
    text=re.sub(" +", " ", text)
    return text