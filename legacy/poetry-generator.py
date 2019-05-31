from wordfreq import word_frequency
from random import sample
import time, re
from itertools import product
import numpy as np

def get_pronunciations(pronunciation_filename='csu-dict-6d.txt',
                       phonemes_filename='phonemes.txt',
                       remove_apostrophe_esses=True,
                       remove_period_words=True,
                       tosort=True):

    with open(pronunciation_filename, 'r+') as f:
        lines = [line.rstrip('\n').split() for line in f]

    pronunciations = [line[1:] for line in lines]
    words = [line[0] for line in lines]

    with open(phonemes_filename, 'r+') as f:
        phonemes_lines = [line.rstrip('\n').split('\t') for line in f]

    vowels = [line[0] for line in phonemes_lines if line[1] == 'vowel']

    phonemes = []
    for line in phonemes_lines:
        if line[1] == 'vowel':
            phonemes += [line[0] + str(i) for i in range(3)]
        else:
            phonemes += [line[0]]

    pronunciations = [
        [syllable+'0' if syllable in vowels else syllable for syllable in line]
        for line in pronunciations
        ]

    assert all([
        all([phoneme in phonemes for phoneme in line])
        for line in pronunciations
        ])

    if tosort:
        word_pronunciation_dict = {
            word:pronunciation for word, pronunciation in zip(words, pronunciations)
        }
        words = sorted(words, key=lambda x: -word_frequency(re.sub("[^a-zA-Z-']|'S", '', x), 'en'))
        pronunciations = list({word:word_pronunciation_dict[word] for word in words}.values())

    def keep_words_pronunciations(words, pronunciations, to_keep):

        words = [words[i] for i in to_keep]
        pronunciations = [pronunciations[i] for i in to_keep]

        return words, pronunciations

    if remove_apostrophe_esses:

        to_keep = [i for i, w in enumerate(words) if len(w) > 2
                   and (w[-2:] != "'S") and ("'S(" not in w)
                   and (w[-2:] != "S'") and ("S'(" not in w)]
        words, pronunciations = keep_words_pronunciations(words, pronunciations, to_keep)

    if remove_period_words:

        to_keep = [i for i, w in enumerate(words) if '.' not in w]
        words, pronunciations = keep_words_pronunciations(words, pronunciations, to_keep)

    return words, pronunciations, phonemes

def get_vowels(pronunciations):

    vowels = [
        [phoneme for phoneme in phonemes if any(char in ['0', '1', '2'] for char in phoneme)]
        for phonemes in pronunciations
    ]

    return vowels

def get_fit_indices(sequences, profile):

    fits = [
        all(sequence_step in profile_step for sequence_step, profile_step  in zip(sequence, profile))
        if len(sequence) <= len(profile) and len(sequence) > 0 else False
        for sequence in sequences
    ]
    fit_indices = [i for i, x in enumerate(fits) if x]

    return fit_indices

def get_fit_information(sequences,
                        profile,
                        sample_number=None,
                        reverse=False
                        ):

    if reverse:
        seqs = [sequence[::-1] for sequence in sequences]
        profile = profile[::-1]

    fit_indices = get_fit_indices(sequences, profile)

    if sample_number is not None:
        fit_indices = sample(fit_indices, max([len(fit_indices), sample_number]))

    new_profiles = [profile[l:] for l in [len(sequences[i]) for i in fit_indices]]
    fit_sequences = [sequences[i] for i in fit_indices]

    if reverse:
        fit_sequences = [sequence[::-1] for sequence in fit_sequences]
        new_profiles = [profile[::-1] for profile in new_profiles]

    return list(zip(fit_indices, fit_sequences, new_profiles))

def categorise_words(pronunciations,
                     foot,
                     length):

    vowel_pronunciations = get_vowels(pronunciations)

    def generate_profile(foot, length, i):

        footlength = len(foot)
        newfoot = foot[i:] + foot[:i]
        profile = newfoot * (length//footlength) + newfoot[:(length)]
        next = (i+length)%footlength

        return profile, next

    categories = product(
        range(len(foot)),
        range(1, length+1)
    )

    category_dict = {}
    for i, wordlength in categories:
        profile, next = generate_profile(foot, wordlength, i)
        word_indices = get_fit_indices(vowel_pronunciations, profile)
        category_dict[(i, wordlength, next)] = word_indices

    return category_dict

def make_readable(word):

    return re.sub('\(.*\)', '', word).lower()

def generate(pronunciations,
             words,
             foot,
             length,
             end,
             lines,
             to_readable = True
             ):

    categorised_words = categorise_words(pronunciations,
                                         foot,
                                         length)

    vpronunciations = get_vowels(pronunciations)
    footlength = len(foot)
    profile = foot * length
    profile_length = len(profile)

    if end is None:

        end_vowels = {vpronunciation[-1] for vpronunciation in vpronunciations}
        end = sample(end_vowels, 1)[0]

    end_words = [(i, seq) for i, seq in enumerate(vpronunciations)
                 if len(seq) > 0
                 and seq[-1] == end and len(seq) <= profile_length
                 and all(phoneme in profile_part for phoneme, profile_part in zip(seq[::-1], profile[::-1]))]

    counts = np.array([len(v) for v in categorised_words.values()])

    output = []
    start_time = time.time()

    while len(output) < lines and (time.time() - start_time) <= 5:

        end_word = sample(end_words, 1)[0]
        new_profile_length = length * footlength - len(end_word[1])
        new_profile = (foot * length)[:new_profile_length]
        start_foot_index = 0
        end_next_foot_index = new_profile_length % footlength

        sequence = []
        phoneme_sequence = []
        current_profile = new_profile

        while len(current_profile) > 0:
            fit = get_fit_information(vpronunciations, current_profile, sample_number=1)
            if len(fit) == 0:
                break
            index, seq, prof = fit[0]
            sequence += [index]
            phoneme_sequence += seq
            current_profile = prof

        if len(phoneme_sequence) == len(new_profile):
            if all(phoneme in set for phoneme, set in zip(phoneme_sequence, new_profile)):
                sequence += [end_word[0]]
                output += [sequence]

    output = [[words[i] for i in seq] for seq in output]

    if to_readable:

        output = [' '.join([make_readable(word) for word in sequence]) for sequence in output]

    return output



def get_pronunciation_end(word, words, pronunciations, vowels, vowel_only=True):
    indices = [i for i, w in enumerate(words) if make_readable(w) == word.lower()]
    index = sample(indices,1)[0]
    pronunciation = pronunciations[index]

    if vowel_only:
        pronunciation_end = get_vowels([pronunciation])[0][-1]
    else:
        vowel_indices = [i for i, phoneme in enumerate(pronunciation) if i in vowels]
        pronunciation_end = pronunciation[vowel_indices[-1]:]

    return pronunciation_end


wo, pr, ph = get_pronunciations(tosort=True)

vowels = {phoneme for phoneme in ph if any(char in ['0', '1', '2'] for char in phoneme)}
stresses = {phoneme for phoneme in ph if any(char in ['1', '2'] for char in phoneme)}
hard_stresses = {phoneme for phoneme in ph if any(char == '1' for char in phoneme)}
unstresses = {phoneme for phoneme in ph if any(char in ['2', '0'] for char in phoneme)}
hard_unstresses = {phoneme for phoneme in ph if any(char == '0' for char in phoneme)}

generate(pronunciations = pr[:10000],
         words = wo[:10000],
         foot = [hard_unstresses, stresses],
         length = 5,
         end = get_pronunciation_end('abound', wo, pr, vowels),
         lines = 10)
