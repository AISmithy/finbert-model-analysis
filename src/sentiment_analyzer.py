from transformers import pipeline

# If Streamlit is available, use it for nicer error reporting in apps. Import
# it conditionally so this module can be used in non-Streamlit contexts.
try:
    import streamlit as st  # type: ignore
    _st_present = True
except Exception:
    st = None  # type: ignore
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
        # Use Streamlit's error UI when available, otherwise fallback to print.
        if _st_present and st is not None:
            try:
                st.error(f"Error loading sentiment model: {e}")
            except Exception:
                # If Streamlit is present but fails for some reason, still print.
                print(f"Error loading sentiment model (streamlit failed): {e}")
        else:
            print(f"Error loading sentiment model: {e}")
        return None


def analyze_sentiment(texts, model=None):
    """
    Analyze sentiment for one or more texts using the provided FinBERT model.

    Usage:
      analyze_sentiment(text, model)
      analyze_sentiment([text1, text2], model)

    This function is tolerant of callers that pass the arguments in either order
    by requiring a pipeline `model` argument; if it's missing we raise a clear
    error.

    Returns a single prediction dict when a single text is provided, or a list
    of prediction dicts when a list is provided.
    """
    # If model was passed as the first argument (legacy), try to detect that.
    # We expect `model` to be callable (pipeline). If the caller passed
    # arguments in the order (model, texts) then `texts` will be the model and
    # `model` will be the actual texts; handle that case.
    if model is None and callable(texts):
        # Caller did analyze_sentiment(model, texts) -> swap
        model, texts = texts, model

    if model is None:
        raise ValueError("No sentiment model provided to analyze_sentiment")

    # Normalize to list for the pipeline call
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    results = model(texts)

    return results[0] if single else results


