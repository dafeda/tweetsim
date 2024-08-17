import streamlit as st
import json
import transformers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STSScorer:
    def __init__(self):
        model_name = 'WillHeld/roberta-base-stsb'
        self._sts_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._sts_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self._sts_model.eval()

    def score(self, sentence1, sentence2):
        sts_tokenizer_output = self._sts_tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt")
        sts_model_output = self._sts_model(**sts_tokenizer_output)
        return sts_model_output['logits'].item()/5

def compare_paragraphs_stsscorer(paragraph1, paragraph2):
    scorer = STSScorer()
    sts_score = scorer.score(paragraph1, paragraph2)
    return sts_score

def load_tweets(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

@st.cache_data
def load_tweets_at_startup():
    json_file_path = 'tweets.json'
    try:
        return load_tweets(json_file_path)
    except FileNotFoundError:
        st.error(f"Error: '{json_file_path}' not found. Please ensure the file exists in the same directory as this script.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: '{json_file_path}' is not a valid JSON file.")
        return None

def check_similarity(new_tweet, tweets_data):
    if not new_tweet:
        st.error("Please enter a tweet.")
        return

    with st.spinner('Calculating similarity scores...'):
        logger.info("Calculating similarity scores")
        scorer = STSScorer()
        similarity_scores = []
        for tweet in tweets_data:
            score = scorer.score(new_tweet, tweet['text'])
            similarity_scores.append((tweet, score))
        # Sort by similarity score in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

    st.subheader("Similarity Results:")
    
    for i, (tweet, score) in enumerate(similarity_scores[:5], 1):  # Show top 5 similar tweets
        st.write(f"{i}. Similar tweet (Score: {score:.2f}):")
        st.write(f"@{tweet['username']}")
        st.write(tweet['text'])
        st.markdown("---")

def main():
    st.title("Tweet Similarity Checker")

    # Load tweets at startup
    tweets_data = load_tweets_at_startup()
    if tweets_data is None:
        return

    new_tweet = st.text_input("Enter a new tweet:", key="new_tweet")
    
    # Check similarity when Enter is pressed or input changes
    if new_tweet:
        check_similarity(new_tweet, tweets_data)

    st.subheader("Tweets generated using llm for testing purposes")
    for tweet in tweets_data:
        st.write(f"@{tweet['username']}")
        st.write(tweet['text'])
        st.markdown("---")  # Add a horizontal line as a separator

if __name__ == "__main__":
    main()