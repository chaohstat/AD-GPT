# 导入库
import os
import unidecode 
import pandas as pd 
import re 
import time 
import nltk 
from nltk.corpus import stopwords 
nltk.download('stopwords') 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from autocorrect import Speller 
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk import word_tokenize 
import string
import pickle as pickle

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is"
    }

def remove_newlines_tabs(text):
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_text
    
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_links(text):
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

def remove_whitespace(text):
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return text

def lower_casing_text(text):
    text = text.lower()
    return text

def replace_hyphens_and_clean_spaces(text):
    """
    Replace single and double hyphens with a single space in a given text,
    then collapse multiple consecutive spaces into a single space.

    Parameters:
    text (str): The text to process.

    Returns:
    str: The processed text with hyphens replaced and unnecessary spaces removed.
    """
    # Replace single and double hyphens with a space
    text = re.sub(r'-{1,2}', ' ', text)

    # Replace multiple consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def split_and_clean_string(input_string):
    """
    Check if a comma exists in the string, split by comma, strip spaces, and return the list of cleaned parts.

    Parameters:
    input_string (str): The string to process.

    Returns:
    list: A list of cleaned, non-empty strings split from the original string.
    """
    # Check if ',' is in the string
    if ',' in input_string:
        # Split the string by ','
        parts = input_string.split(',')
        # Remove any leading/trailing whitespace from each part and filter out empty strings
        cleaned_parts = [part.strip() for part in parts if part.strip()]
        return cleaned_parts
    else:
        # If no comma is present, return the original string in a list after stripping whitespace
        return [input_string.strip()] if input_string.strip() else []

def extract_and_split_by_parentheses(input_string):
    result = []
    
    # Find all occurrences of text within parentheses
    inside_parentheses = re.findall(r'\(([^)]+)\)', input_string)
    
    # Add extracted contents inside parentheses to the result list
    result.extend(inside_parentheses)
    
    # Remove text within parentheses from the original string and split into parts
    # We replace the text inside parentheses with a placeholder to keep split sections in order.
    temp_string = re.sub(r'\([^)]*\)', '{}', input_string)
    parts_without_parentheses = [part.strip() for part in temp_string.split('{}') if part.strip()]
    
    # Add non-parenthetical parts to the result list
    result.extend(parts_without_parentheses)
    
    return result

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    list_Of_tokens = text.split(' ')

    for Word in list_Of_tokens: 
         if Word in CONTRACTION_MAP: 
                list_Of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_Of_tokens]
                
    String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens) 
    return String_Of_tokens

def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

print(repr(remove_text_inside_brackets(
    "This is a sentence. (once a day) [twice a day]")))
# -> 'This is a sentence.  '

def remove_characters_regex(text):
    return re.sub(r'[{}?]', '', text)


def truncate_string(value, max_length=3000, suffix='.'):
    if value == 'NA':
        return 'unknown'
    else:
        string_value = str(value)
        string_truncated = string_value[:min(len(string_value), (max_length - len(suffix)))]
        suffix = (suffix if len(string_value) > max_length else '')
    return (string_truncated+suffix).strip().strip('.')

def find_text_after_keyword(text, keyword):
    pattern = re.escape(keyword) + r'(.*)'
    # Search for the pattern in the text
    match = re.search(pattern, text)
    # If a match is found, return the group that follows the keyword
    if match:
        return match.group(1).strip()
    else:
        return ""

def remove_text_in_brackets(text):
    pattern = r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}'
    # Continuously apply the regex to handle nested brackets
    while re.search(pattern, text):
        text = re.sub(pattern, '', text)

    return text

def basic_clean_pipe(text):
    CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    }
    text = remove_text_inside_brackets(text)
    text = remove_newlines_tabs(text)
    text = strip_html_tags(text)
    text = remove_links(text)
    text = remove_whitespace(text)
    text = lower_casing_text(text)
    text = expand_contractions(text, CONTRACTION_MAP)

    text = truncate_string(text, max_length=5000)

    return text

def find_text_after_keyword(text, keyword):
    pattern = re.escape(keyword) + r'(.*)'
    # Search for the pattern in the text
    match = re.search(pattern, text)
    # If a match is found, return the group that follows the keyword
    if match:
        return match.group(1).strip()
    else:
        return ""
    
def extract_words_between(text, start_word, end_word):
    pattern = fr'{re.escape(start_word)}(.*?){re.escape(end_word)}'
    
    # Searching for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)

    # If a match is found, return the group of words between the start and end words
    if match:
        return match.group(1).strip()
    else:
        return None

def remove_text_in_brackets(text):
    pattern = r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}'
    # Continuously apply the regex to handle nested brackets
    while re.search(pattern, text):
        text = re.sub(pattern, '', text)

    return text

def read_file_as_string(file_path):
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " does not exist")
    all_the_text = open(file_path).read()
    #print(type(all_the_text))
    return all_the_text

def find_and_return_matched_forms(text, base_word):
    normalized_base_word = re.sub(r'-{1,2}|\s', '-', base_word)

    # Create a regex pattern that matches all variations of separators between parts
    parts = re.split(r'-', normalized_base_word)  # Split on hyphen to get parts
    regex_form = r'\b' + r'[-\s]*'.join(map(re.escape, parts)) + r'(s|es)?\b'

    # Perform a case-insensitive search to find all matches
    matches = re.findall(regex_form, text, re.IGNORECASE)

    # Return the list of matched words
    return matches


def generate_term_geneprotein(gene_name, g_sum, pp, token_seeker):
    instruction_set = {#'i0':'Generate a summary of the phenotype {}.',
                       'i1':'gene {} coding protein {}, given function of gene {}, infer function of protein {}.',
                       'i2':'protein {} is the product of gene {}, given function of protein {}, conclude function of gene {}.',
                      }
    relation_s = ['are related', 'are associated']
    terms_ = []
    gene_name_new = gene_name#.replace(' ', '-')
    pp_name = pp[0]
    pp_name_new = pp_name#.replace(' ', '-')
    token_seeker.append({'gene': gene_name_new, 'protein':pp_name_new})
    for i, v in instruction_set.items():
        print(i)
        if i == 'i1':
            term_dict = {} #OrderedDict()
            term_dict["instruction"] = instruction_set['i1'].format(gene_name_new, pp_name_new, gene_name_new, pp_name_new)
            input_string = "gene function summary is: " + basic_clean_pipe(g_sum)
            output_string = "inferred function summary of thge protein {} is: ".format(pp_name_new) + basic_clean_pipe(pp[1])
            term_dict["input"] = input_string.replace('\n', '')
            term_dict["output"] = output_string.replace('\n', '')
            terms_.append(term_dict)
        elif i == 'i2':
            term_dict = {} #OrderedDict()
            term_dict["instruction"] = instruction_set['i2'].format(pp_name_new, gene_name_new, pp_name_new, gene_name_new)
            input_string = "protein function summary is: " + basic_clean_pipe(pp[1])
            output_string = "inferred function summary of the gene {} is: ".format(gene_name_new) + basic_clean_pipe(g_sum)
            term_dict["input"] = input_string.replace('\n', '')
            term_dict["output"] = output_string.replace('\n', '')
            terms_.append(term_dict)
        else:
            continue
    return terms_

def generate_term_genetrait(gene_name, g_sum, omim_saves, token_seeker):
    instruction_set = {'i0':'Generate a summary of the phenotype {}.',
                       'i5':'Given the summary of a gene {} and molecular summary of phenotype {}, identify the relationship of {} and {}.',
                       'i7':'Given the summary of a gene {} and inheritance summary of phenotype {}, identify the relationship of {} and {}.',
                      }
    relation_s = ['are related', 'are associated']
    terms_ = []
    gene_name_new = gene_name#.replace(' ', '-')
    phe_name = omim_saves['title']
    phe_name_new = phe_name.strip('-').replace(',', '')
    token_seeker.append({'gene': gene_name_new, 'trait':phe_name_new})
    for i, v in instruction_set.items():
        print(i)
        if i == 'i5':
            if omim_saves['molelular'] != 'NA' and len(omim_saves['molelular']) > 10:
                term_dict = {} #OrderedDict()
                term_dict["instruction"] = instruction_set['i5'].format(gene_name_new, phe_name_new, gene_name_new, phe_name_new)
                input_string = "Gene function summary is: " + basic_clean_pipe(g_sum)
                input_string += " Phenotype Molecular summary is " + basic_clean_pipe(omim_saves['molelular']) 
                for r in relation_s:
                    output_string = "Phenotype {} and Gene {} is {}.".format(gene_name, omim_saves['title'], r)
                    term_dict["input"] = input_string.replace('\n', '')
                    term_dict["output"] = output_string.replace('\n', '')
                    terms_.append(term_dict)


        elif i == 'i7':
            if omim_saves['inherit'] != 'NA' and len(omim_saves['inherit']) > 10:
                term_dict = {} #OrderedDict()
                term_dict["instruction"] = instruction_set['i5'].format(gene_name_new, phe_name_new, gene_name_new, phe_name_new)
                input_string = "Gene function summary is: " + basic_clean_pipe(g_sum)
                input_string += " Phenotype inheritance summary is " + basic_clean_pipe(omim_saves['molelular']) 
                for r in relation_s:
                    output_string = "Phenotype {} and Gene {} is {}.".format(gene_name, omim_saves['title'], r)
                    term_dict["input"] = input_string.replace('\n', '')
                    term_dict["output"] = output_string.replace('\n', '')
                    terms_.append(term_dict)
        else:
            continue
    #print(terms_)
    return terms_, 



























