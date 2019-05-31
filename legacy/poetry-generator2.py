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

stresscheck = {
    'hardstress' : check_hardstress,
    'stress' : check_stress,
    'hardnostress' : check_hardnostress,
    'nostress' : check_nostress,
    'any' : check_vowel
}

def generate(pronunciations,
             words,
             foot,
             length,
             lines,
             rhyme_profile,
             match_consonants=True,
             multi_word=True,
             to_readable = True
             ):

    categorised_words = categorise_words(pronunciations,
                                         foot,
                                         length)

    vpronunciations = get_vowels(pronunciations)
    footlength = len(foot)
    profile = foot * length
    profile_length = len(profile)

    end_words = get_rhymes(line_profile=foot*length,
                           rhyme_profile=rhyme_profile,
                           words=words,
                           pronunciations=pronunciations,
                           match_consonants=match_consonants)

    counts = np.array([len(v) for v in categorised_words.values()])

    output = []
    start_time = time.time()

    while len(output) < lines and (time.time() - start_time) <= 5:

        end_word = sample(end_words, 1)[0]
        end_word_pronunciation = pronunciations[end_word]
        end_word_vowel_pronunciation = get_vowels([end_word_pronunciation])[0]
        new_profile_length = length * footlength - len(end_word_vowel_pronunciation)
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
                sequence += [end_word]
                output += [sequence]

    output = [[words[i] for i in seq] for seq in output]

    if to_readable:

        output = [' '.join([make_readable(word) for word in sequence]) for sequence in output]

    return output

def get_pronunciation_end(word, words, pronunciations, full_rhyme=True):

    indices = [i for i, w in enumerate(words) if make_readable(w) == word.lower()]
    index = sample(indices,1)[0]
    pronunciation = pronunciations[index]

    if full_rhyme:
        hard_unstress_indices = [i for i, p in enumerate(pronunciation) if any(char == '0' for char in p)]
        stress_indices = [i for i, p in enumerate(pronunciation) if any(char in ['1', '2'] for char in p)]
        if len(hard_unstress_indices) == 0:
            pronunciation_end = pronunciation[stress_indices[-1]:]
        if len(hard_unstress_indices) > 0:
            if len(stress_indices) == 0:
                pronunciation_end = pronunciation
            else:
                first_rhyme_syllable = min(stress_indices[-1], hard_unstress_indices[-1])
                pronunciation_end = pronunciation[first_rhyme_syllable:]
    else:
        vowel_indices = [i for i, p in enumerate(pronunciation) if any(char in ['0', '1', '2'] for char in p)]
        pronunciation_end = pronunciation[vowel_indices[-1]:]

    return pronunciation_end

def get_rhymes(line_profile, rhyme_profile, words, pronunciations, match_consonants=True, multi_word=True):

    vowel_rhyme_profile = [phoneme for phoneme in rhyme_profile if stresscheck['any'](phoneme)]
    vowel_pronunciations = [
        [phoneme for phoneme in pronunciation if stresscheck['any'](phoneme)]
        for pronunciation in pronunciations
        ]
    assert all(rhyme_vowel in line_vowels
               for line_vowels, rhyme_vowel
               in zip(line_profile[::-1], vowel_rhyme_profile[::-1]))


    word_limit = 1 if (not multi_word) or (len(vowel_rhyme_profile) == 1) else 2

    word_indices = []

    if not match_consonants:
        pronunciations = [
            [phoneme for phoneme in pronunciation if stresscheck['any'](phoneme)]
            for pronunciation in pronunciations
            ]
        rhyme_profile = [phoneme for phoneme in rhyme_profile if stresscheck['any'](phoneme)]

    single_word_rhyme_anyfit_indices = [
        i for i, pronunciation in enumerate(pronunciations)
        if all(pronunciation_phoneme == rhyme_phoneme
               for pronunciation_phoneme, rhyme_phoneme
               in zip(pronunciation[::-1], rhyme_profile[::-1]))
    ]

    single_word_line_anyfit_indices = [
        i for i, pronunciation in enumerate(vowel_pronunciations)
        if all(pronunciation_phoneme in line_phoneme
               for pronunciation_phoneme, line_phoneme
               in zip(pronunciation[::-1], line_profile[::-1]))
    ]

    single_word_anyfit_indices = [sequence for sequence in single_word_rhyme_anyfit_indices
                                  if sequence in single_word_line_anyfit_indices]

    single_word_fullfit_indices = [
        fit for fit in single_word_anyfit_indices
        if len(pronunciations[fit]) >= len(rhyme_profile)
    ]

    word_indices += single_word_fullfit_indices

    return word_indices





wo, pr, ph = get_pronunciations(tosort=True)

vowels = {phoneme for phoneme in ph if stresscheck['any'](phoneme)}
stresses = {phoneme for phoneme in ph if stresscheck['stress'](phoneme)}
hard_stresses = {phoneme for phoneme in ph if stresscheck['hardstress'](phoneme)}
nostresses = {phoneme for phoneme in ph if stresscheck['nostress'](phoneme)}
hard_nostresses = {phoneme for phoneme in ph if stresscheck['hardnostress'](phoneme)}

get_vowels([get_pronunciation_end('hesitation', wo, pr)])

rhymes = get_rhymes([stresses, nostresses]*5,get_pronunciation_end('slayer', wo, pr) , wo, pr, match_consonants=True)
sample([wo[i] for i in rhymes], 10)

iamb = [hard_nostresses, stresses]
trochee = [stresses, hard_nostresses]
wut = [stresses, hard_nostresses, hard_nostresses]
trochee2 = [stresses, nostresses]
generate(pronunciations=pr[:50000],
         words=wo[:50000],
         foot=trochee,
         length=5,
         lines=10,
         rhyme_profile=get_pronunciation_end('fountain', wo, pr, full_rhyme=False),
         match_consonants=True,
         multi_word=True,
         to_readable = True
         )
get_pronunciation_end('fountain', wo, pr)
