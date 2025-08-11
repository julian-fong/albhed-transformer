from transformers import GPT2Tokenizer

def get_tokenizer():
    """
    Returns the tokenizer for the GPT-2 model.
    
    The tokenizer is GPT2 Tokenizer from the transformers library with two extra special tokens:
    - <|pad|> for padding -> 50257
    
    Returns
    -------
    
    tokenizer : GPT2Tokenizer
    """
    # gpt2 - 50257 tokens + 1 special tokens = 50258
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("./models/tokenizer/") # try local first
        return tokenizer
    except Exception:
        print("Error loading tokenizer from local path, will load fresh and save locally")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # download if not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.save_pretrained("./models/tokenizer/")
        return tokenizer
