import html

from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from modules import script_callbacks, shared

import gradio as gr

def validate_reconstructed(clip, vocab, subword, word):
    token = subword + word
    token_ids = clip.tokenizer(token, truncation=False, add_special_tokens=False)["input_ids"]
    reconstructed = [vocab.get(x, "") for x in token_ids]
    return len(reconstructed) == 2 and reconstructed[0] == subword and reconstructed[1][:-4] == word

def generate(text):
    clip: FrozenCLIPEmbedder = shared.sd_model.cond_stage_model.wrapped
    vocab = {v: k for k, v in clip.tokenizer.get_vocab().items()}
    special_characters = '0123456789!@#$%^&*()[]{}|~-+?_=,<>/\\"\'`.:;'
    special_strings = [ 'bbc', 'cnn', 'cbs', 'pnr', 'nbc', 'mbc', 'msnbc', 'sbs', 'shh', 'lgb', 'lgbt', 'lgbtq', 'blm', 'mr', 'mrs', 'ms', 'nfl', 'nhl', 'bbq', 'gps', 'rgb', 'png', 'jpg', 'bmp', 'mpg', 'nfc', 'php', 'dns', 'sql', 'bmw', 'fml', 'wtf', 'dvd', 'vhs', 'rts', 'dnd', 'rpg', 'fps', 'ftl', 'www', 'http', 'https', 'tnt', 'cpr', 'ctv', 'phd', 'std', 'tsp', 'sks', 'nsfw', 'sfw']
    
    starts = []
    ends = []
    
    for tokenText in vocab.values():
        if tokenText.endswith("</w>"):
            tokenText = tokenText[:-4]
            tokenList = ends
        else:
            tokenList = starts
        
        if len(tokenText) > 1 and not any(c in special_characters for c in tokenText) and not tokenText in special_strings:
            tokenList.append(tokenText)
    
    # prioritize short text and least vowels to reduce conflicts with existing words https://stackoverflow.com/a/21115970
    starts.sort(key = lambda word: 6 * sum(ch in 'aeiouy' for ch in word) + len(word))
    ends.sort(key = lambda word: 6 * sum(ch in 'aeiouy' for ch in word) + len(word))
    
    #print(f'starts: {starts}')
    #print(f'ends: {ends}')
    
    mappings = {}
    
    for phrase in text.split(","):
        phrase = phrase.strip()
        if len(phrase) == 0:
            continue
        
        startI = 0
        endI = 0
        while not validate_reconstructed(clip, vocab, starts[startI], ends[endI]):
            if startI < endI:
                startI += 1
            else:
                endI += 1
        
        token = starts.pop(startI) + ends.pop(endI)
        
        mappings[phrase] = token
    
    return mappings


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        words = gr.Textbox(show_label=False, lines=8, placeholder="word1, word2, phrase 1, word3, ...")
        go = gr.Button(value="Generate", variant="primary")
        result = gr.Textbox(show_label=False, lines=8)

        go.click(
            fn=generate,
            inputs=[words],
            outputs=[result],
        )
    
    return [(ui, "Finetuning Tokens", "finetuning_tokens")]


script_callbacks.on_ui_tabs(add_tab)
