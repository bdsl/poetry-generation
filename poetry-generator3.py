from wordfreq import word_frequency
from random import sample
import time
import re
from itertools import product
import numpy as np


def get_raw_dictionary(pronunciation_filename='csu-dict-6d.txt'):

    with open(pronunciation_filename, 'r+') as f:
        lines = [line.rstrip('\n').split() for line in f]

    dictionary = [{
        'word': line[0],
        'pronunciation': line[1:]
    } for line in lines]

    return dictionary


def get_phonemes(phonemes_filename='phonemes.txt',
                 return_vowel_bases=True):

    with open(phonemes_filename, 'r+') as f:
        phonemes_lines = [line.rstrip('\n').split('\t') for line in f]

    vowel_bases = [line[0] for line in phonemes_lines if line[1] == 'vowel']

    phonemes = []
    for line in phonemes_lines:
        if line[1] == 'vowel':
            phonemes += [line[0] + str(i) for i in range(3)]
        else:
            phonemes += [line[0]]
    if return_vowel_bases:
        return phonemes, vowel_bases
    else:
        return phonemes


def clean_pronunciations(dictionary,
                         phonemes,
                         vowel_bases):
    dictionary = [{
        'word': entry['word'],
        'pronunciation': [
            syllable + '0' if syllable in vowel_bases else syllable
            for syllable in entry['pronunciation']
        ]}
        for entry in dictionary
    ]

    assert all([
        all([phoneme in phonemes for phoneme in pronunciation])
        for pronunciation in [entry['pronunciation'] for entry in dictionary]
    ])

    return dictionary


def prune_entries(dictionary,
                  remove_apostophe_esses=True,
                  remove_period_words=True):

    if remove_apostophe_esses:
        dictionary = [
            entry for entry in dictionary
            if len(entry['word']) > 2 and
            (entry['word'][-2:] != "'S") and ("S(" not in entry['word']) and
            (entry['word'][-2:] != "S'") and ("S'(" not in entry['word'])
        ]

    if remove_period_words:
        dictionary = [
            entry for entry in dictionary
            if '.' not in entry['word']
        ]

    return dictionary


def sort_entries(dictionary):

    def get_word_base(word):
        return re.sub("[^a-zA-Z-']|'S", "", word)

    dictionary = sorted(dictionary,
                        key=lambda x: -word_frequency(get_word_base(x['word']), 'en'))

    return dictionary


def get_vowels(dictionary):

    dictionary = [{
        'word': entry['word'],
        'pronunciation': entry['pronunciation'],
        'vowels': [phoneme for phoneme in entry['pronunciation']
                   if any(char in ['0', '1', '2'] for char in phoneme)]
    } for entry in dictionary]

    return dictionary


def get_dictionary(pronunciation_filename='csu-dict-6d.txt',
                   phonemes_filename='phonemes.txt',
                   remove_apostrophe_esses=True,
                   remove_period_words=True,
                   freqsort=True):

    dictionary = get_raw_dictionary(pronunciation_filename)

    phonemes, vowel_bases = get_phonemes(phonemes_filename)

    dictionary = clean_pronunciations(dictionary,
                                      phonemes,
                                      vowel_bases)

    dictionary = prune_entries(dictionary,
                               remove_apostrophe_esses,
                               remove_period_words)

    dictionary = get_vowels(dictionary)

    if freqsort:
        dictionary = sort_entries(dictionary)

    return dictionary


def check_hardstress(phoneme):
    return any(char == '1' for char in phoneme)


def check_stress(phoneme):
    return any(char in ['1', '2'] for char in phoneme)


def check_hardnostress(phoneme):
    return any(char == '0' for char in phoneme)


def check_nostress(phoneme):
    return any(char in ['0', '2'] for char in phoneme)


def check_vowel(phoneme):
    return any(char in ['0', '1', '2'] for char in phoneme)


checkstress = {
    'hardstress': check_hardstress,
    'stress': check_stress,
    'hardnostress': check_hardnostress,
    'nostress': check_nostress,
    'any': check_vowel
}

dictionary = get_dictionary()
phonemes = get_phonemes(return_vowel_bases=False)
vowels = {phoneme for phoneme in phonemes if checkstress['any'](phoneme)}
stresses = {phoneme for phoneme in phonemes if checkstress['stress'](phoneme)}
hard_stresses = {
    phoneme for phoneme in phonemes if checkstress['hardstress'](phoneme)}
nostresses = {
    phoneme for phoneme in phonemes if checkstress['nostress'](phoneme)}
hard_nostresses = {
    phoneme for phoneme in phonemes if checkstress['hardnostress'](phoneme)}


def get_fit_indices(dictionary, profile):

    fit_indices = [
        i for i, entry in enumerate(dictionary)
        if len(entry['vowels']) <= len(profile) and
        len(entry['vowels']) > 0 and
        all(vowel in profile_step for vowel, profile_step
            in zip(entry['vowels'], profile))
    ]

    return fit_indices


def fill_profile(profile, dictionary):

    current_profile = profile
    filled_profile = []

    def get_filled_profile_vowels():
        return [vowel
                for entry in filled_profile
                for vowel in entry['vowels']]

    def get_new_profile():
        return profile[len(get_filled_profile_vowels()):]

    valid_lengths = set(range(len(profile)))

    while len(current_profile) > 0:
        filled_profile_vowels = get_filled_profile_vowels()
        filled_profile_vowels_length = len(filled_profile_vowels)
        fit_indices = get_fit_indices(dictionary, profile=current_profile)
        fit_indices = [fit for fit in fit_indices
                       if (filled_profile_vowels_length + len(dictionary[fit]['vowels']))
                       in (list(valid_lengths) + [len(profile)])]
        if len(fit_indices) == 0:
            set(valid_lengths).remove(filled_profile_vowels_length)
            del filled_profile[-1]
            current_profile = get_new_profile()
            continue
        fits = [{'index': i, 'vowels': dictionary[i]['vowels']}
                for i in fit_indices]
        filled_profile.append(sample(fits, 1)[0])
        current_profile = get_new_profile()

    if len(filled_profile) == 0:
        print('no fill found')
        return

    filled_profile = [part['index'] for part in filled_profile]

    return filled_profile

def get_rhymes(dictionary,
               line_profile,
               rhyme_profile,
               match_consonants):

    vowel_rhyme_profile = [
        phoneme for phoneme in rhyme_profile if checkstress['any'](phoneme)]
    assert all(rhyme_vowel in line_vowels
               for line_vowels, rhyme_vowel
               in zip(line_profile[::-1], vowel_rhyme_profile[::-1]))

    matchtype = 'pronunciation' if match_consonants else 'vowels'

    single_word_fit_indices = [
        i for i, entry in enumerate(dictionary)
        if all(pronunciation_phoneme == rhyme_phoneme
               for pronunciation_phoneme, rhyme_phoneme
               in zip(entry[matchtype][::-1], rhyme_profile[::-1])) and
        len(entry['vowels']) >= len(vowel_rhyme_profile) and
        all(pronunciation_phoneme in line_phoneme
            for pronunciation_phoneme, line_phoneme
            in zip(entry['vowels'][::-1], line_profile[::-1]))
    ]

    return single_word_fit_indices

def make_readable(word):
    return re.sub('\(.*\)', '', word).lower()

def get_pronunciation_end(word, dictionary, full_rhyme=True):

    indices = [i for i, entry in enumerate(dictionary) if make_readable(entry['word']) == word.lower()]
    pronunciation_ends = []

    if full_rhyme:
        for i in indices:
            pronunciation = dictionary[i]['pronunciation']
            hard_nostress_indices = [i for i, phoneme
                                     in enumerate(pronunciation)
                                     if checkstress['hardnostress'](phoneme)]
            stress_indices = [i for i, phoneme
                              in enumerate(pronunciation)
                              if checkstress['stress'](phoneme)]
            if len(hard_nostress_indices) == 0:
                pronunciation_end = pronunciation[stress_indices[-1]:]
            if len(hard_nostress_indices) > 0:
                if len(stress_indices) == 0:
                    pronunciation_end = pronunciation
                else:
                    first_rhyme_syllable = min(stress_indices[-1], hard_nostress_indices[-1])
                    pronunciation_end = pronunciation[first_rhyme_syllable:]
            pronunciation_ends += [pronunciation_end]
    else:
        for i in indices:
            pronunciation = dictionary[i]['pronunciation']
            vowel_indices = [i for i, phoneme in enumerate(pronunciation)
                             if stresscheck['any'](phoneme)]
            pronunciation_end = pronunciation[vowel_indices[-1]:]
            pronunciation_ends += [pronunciation_end]

    return pronunciation_ends

dictionary = get_dictionary()

def generate(dictionary,
             line_profile,
             rhyme_profile,
             linenum,
             match_consonants=True,
             full_rhyme=True,
             to_readable=True):

    profile_length = len(line_profile)

    end_indices = get_rhymes(dictionary=dictionary,
                             line_profile=line_profile,
                             rhyme_profile=rhyme_profile,
                             match_consonants=match_consonants)

    output = []
    start_time = time.time()
    while (len(output) < linenum) and (time.time() - start_time) < 5:
        end_index = sample(end_indices, 1)[0]
        end_entry = dictionary[end_index]
        new_profile_length = profile_length - len(end_entry['vowels'])
        new_profile = line_profile[:new_profile_length]

        start_indices = fill_profile(new_profile, dictionary)
        filled_profile = start_indices + [end_index]
        output += [[dictionary[i] for i in filled_profile]]

    if to_readable:
        output = [[make_readable(entry['word']) for entry in line]
                  for line in output]

    return output

poem = generate(dictionary[:30000],
         [stresses, nostresses]*4,
         get_pronunciation_end('traitor', dictionary)[0],
         5)

print ('\n'.join(map(lambda x: ' '.join(x), poem)))