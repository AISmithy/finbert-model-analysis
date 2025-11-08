from transformers import pipeline

_st_present = False

def load_sentiment_model():
    """
    Loads the FinBERT sentiment analysis model.
    """
    try:
        # Load the specialized financial sentiment model
        model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        return model
    except Exception as e:
        if _st_present:
            st.error(f"Error loading sentiment model: {e}")
        else:
            print(f"Error loading sentiment model: {e}")
        return None


