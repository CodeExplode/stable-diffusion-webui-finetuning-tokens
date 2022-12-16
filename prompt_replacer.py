# this isn't part of the extension though is an example of how to use the results programatically, and can be slotted into any SD build
# call replace_prompt below in whatever training/inference implementation you are using where it takes the prompt as input, e.g. prompt = replace_prompt(prompt)
# paste results from the finetuning-tokens extension below in mapping, or write manual mappings
# if as_tags is set to True, only occurances which exist as a comma-separated tag will be replaced. e.g. in prompt 'word1, word1 word2, word3', only the first word1 would be replaced
# if as_tags is False, any occurance of the text will be replaced, such as in a sentence (e.g. 'a photo of word1 standing on a street', word1 would be replaced)

mapping = {} # example: mapping = {'word1': 'thst', 'word2': 'stts', 'a phrase 1': 'chss', 'word3': 'shch', 'a phrase 3': 'whds'}

def replace_prompt(prompt, as_tags=False):
    if as_tags:
        prompt = prompt.split(",")
    
    for word, token in mapping.items():
        if as_tags:
            prompt[:] = [x if x.strip() != word else token for x in prompt] # https://stackoverflow.com/a/24201988
        else:
            prompt = prompt.replace(word, token)
    
    if as_tags:
        prompt = ','.join(prompt)
        
    return prompt