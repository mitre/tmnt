# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


import unicodedata
import re
import io

__all__ = ['BasicTokenizer']

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, use_stop_words=True, custom_stop_word_file=None, encoding='utf-8'):
        self.encoding = encoding
        self.do_lower_case = do_lower_case
        self.use_stop_words = use_stop_words
        self.stop_word_set = \
            self.get_stop_word_set(custom_stop_word_file) if custom_stop_word_file is not None else default_stop_words
        self.num_re = re.compile('[-+]?[.\d]*[\d]+[:,.\d]*$') ## matches straight number
        
    def get_stop_word_set(self, f):
        wds = []
        with io.open(f, 'r', encoding=self.encoding) as fp:
            for w in fp:
                wds.append(w)
        return set(wds)

    def __call__(self, text):
        return self.tokenize(text)

    def to_unicode(self, text):
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self.to_unicode(text)
        text = self._clean_text(text)
        if self.do_lower_case:
            text = self._run_strip_accents(text.lower())
        orig_tokens = self.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(self._run_split_on_punc(token, keep_punct=False))
        if self.use_stop_words:
            final_tokens = [t for t in split_tokens if len(t) > 1 and not t in self.stop_word_set and not self.num_re.match(t)]
        else:
            final_tokens = [t for t in split_tokens if len(t) > 1 and not self.num_re.match(t)]
        output_tokens = self.whitespace_tokenize(' '.join(final_tokens))
        return output_tokens

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text, keep_punct=True):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                if keep_punct:
                    output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char in [' ', '\t', '\n', '\r']:
            return True
        cat = unicodedata.category(char)
        if cat == 'Zs':
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        if char in ['\t', '\n', '\r']:
            return False
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            return True
        return False


    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        group0 = cp >= 33 and cp <= 47
        group1 = cp >= 58 and cp <= 64
        group2 = cp >= 91 and cp <= 96
        group3 = cp >= 123 and cp <= 126
        if (group0 or group1 or group2 or group3):
            return True
        cat = unicodedata.category(char)
        if cat.startswith('P') or cat.startswith('S') or cat.startswith('C') or cat.startswith('M'):
            return True
        return False

default_stop_words = \
    set([
        'able',
        'about',
        'above',
        'according',
        'accordingly',
        'across',
        'actually',
        'after',
        'afterwards',
        'again',
        'against',
        'ago',
        'all',
        'allow',
        'allows',
        'almost',
        'alone',
        'along',
        'already',
        'also',
        'although',
        'always',
        'am',
        'among',
        'amongst',
        'amp',
        'an',
        'and',
        'another',
        'any',
        'anybody',
        'anyhow',
        'anyone',
        'anything',
        'anyway',
        'anyways',
        'anywhere',
        'apart',
        'appear',
        'appreciate',
        'appropriate',
        'are',
        'around',
        'as',
        'aside',
        'ask',
        'asking',
        'associated',
        'at',
        'available',
        'away',
        'awfully',
        'be',
        'became',
        'because',
        'become',
        'becomes',
        'becoming',
        'been',
        'before',
        'beforehand',
        'behind',
        'being',
        'believe',
        'below',
        'beside',
        'besides',
        'best',
        'better',
        'between',
        'beyond',
        'both',
        'brief',
        'but',
        'by',
        'came',
        'can',
        'cannot',
        'cant',
        'cause',
        'causes',
        'certain',
        'certainly',
        'changes',
        'clearly',
        'co',
        'com',
        'come',
        'comes',
        'concerning',
        'consequently',
        'consider',
        'considering',
        'contain',
        'containing',
        'contains',
        'corresponding',
        'could',
        'couldn',
        'course',
        'currently',
        'definitely',
        'described',
        'despite',
        'did',
        'didn',
        'different',
        'do',
        'does',
        'doesn',
        'doing',
        'done',
        'down',
        'downwards',
        'during',
        'each',
        'edu',
        'eg',
        'eight',
        'either',
        'else',
        'elsewhere',
        'en',
        'enough',
        'entirely',
        'especially',
        'et',
        'etc',
        'even',
        'ever',
        'every',
        'everybody',
        'everyone',
        'everything',
        'everywhere',
        'ex',
        'exactly',
        'example',
        'except',
        'far',
        'few',
        'fifth',
        'first',
        'five',
        'followed',
        'following',
        'follows',
        'for',
        'former',
        'formerly',
        'forth',
        'four',
        'from',
        'further',
        'furthermore',
        'get',
        'gets',
        'getting',
        'given',
        'gives',
        'go',
        'goes',
        'going',
        'gone',
        'got',
        'gotten',
        'greetings',
        'gt',
        'had',
        'happens',
        'hardly',
        'has',
        'have',
        'having',
        'he',
        'hello',
        'help',
        'hence',
        'her',
        'here',
        'hereafter',
        'hereby',
        'herein',
        'hereupon',
        'hers',
        'herself',
        'hi',
        'him',
        'himself',
        'his',
        'hither',
        'hopefully',
        'how',
        'howbeit',
        'however',
        'http',
        'https',
        'ie',
        'if',
        'ignored',
        'immediate',
        'in',
        'inasmuch',
        'inc',
        'indeed',
        'indicate',
        'indicated',
        'indicates',
        'inner',
        'insofar',
        'instead',
        'into',
        'inward',
        'is',
        'it',
        'its',
        'itself',
        'just',
        'keep',
        'keeps',
        'kept',
        'know',
        'knows',
        'known',
        'last',
        'lately',
        'later',
        'latter',
        'latterly',
        'least',
        'less',
        'lest',
        'let',
        'like',
        'liked',
        'likely',
        'little',
        'lol',
        'look',
        'looking',
        'looks',
        'lt',
        'ltd',
        'll',
        'mainly',
        'many',
        'may',
        'maybe',
        'me',
        'mean',
        'meanwhile',
        'merely',
        'might',
        'more',
        'moreover',
        'most',
        'mostly',
        'mr',
        'mrs',
        'much',
        'must',
        'my',
        'myself',
        'name',
        'namely',
        'nd',
        'near',
        'nearly',
        'necessary',
        'need',
        'needs',
        'neither',
        'never',
        'nevertheless',
        'new',
        'next',
        'nine',
        'no',
        'nobody',
        'non',
        'none',
        'noone',
        'nor',
        'normally',
        'not',
        'nothing',
        'novel',
        'now',
        'nowhere',
        'obviously',
        'of',
        'off',
        'often',
        'oh',
        'ok',
        'okay',
        'old',
        'on',
        'once',
        'one',
        'ones',
        'only',
        'onto',
        'or',
        'org',
        'other',
        'others',
        'otherwise',
        'ought',
        'our',
        'ours',
        'ourselves',
        'out',
        'outside',
        'over',
        'overall',
        'own',
        'particular',
        'particularly',
        'per',
        'perhaps',
        'placed',
        'please',
        'plus',
        'possible',
        'presumably',
        'probably',
        'provides',
        'que',
        'quite',
        'qv',
        'rather',
        'rd',
        're',
        'really',
        'reasonably',
        'regarding',
        'regardless',
        'regards',
        'relatively',
        'respectively',
        'right',
        'rt',
        'said',
        'same',
        'saw',
        'say',
        'saying',
        'says',
        'second',
        'secondly',
        'see',
        'seeing',
        'seem',
        'seemed',
        'seeming',
        'seems',
        'seen',
        'self',
        'selves',
        'sensible',
        'sent',
        'serious',
        'seriously',
        'seven',
        'several',
        'shall',
        'she',
        'should',
        'shouldn',
        'since',
        'six',
        'so',
        'some',
        'somebody',
        'somehow',
        'someone',
        'something',
        'sometime',
        'sometimes',
        'somewhat',
        'somewhere',
        'soon',
        'sorry',
        'specified',
        'specify',
        'specifying',
        'still',
        'sub',
        'such',
        'sup',
        'sure',
        'take',
        'taken',
        'tell',
        'tends',
        'th',
        'than',
        'thank',
        'thanks',
        'thanx',
        'that',
        'thats',
        'the',
        'their',
        'theirs',
        'them',
        'themselves',
        'then',
        'thence',
        'there',
        'thereafter',
        'thereby',
        'therefore',
        'therein',
        'theres',
        'thereupon',
        'these',
        'they',
        'think',
        'third',
        'this',
        'thorough',
        'thoroughly',
        'those',
        'though',
        'three',
        'through',
        'throughout',
        'thru',
        'thus',
        'to',
        'together',
        'too',
        'took',
        'toward',
        'towards',
        'tried',
        'tries',
        'truly',
        'try',
        'trying',
        'twice',
        'two',
        'un',
        'under',
        'unfortunately',
        'unless',
        'unlikely',
        'until',
        'unto',
        'up',
        'upon',
        'us',
        'use',
        'used',
        'useful',
        'uses',
        'using',
        'usually',
        'uucp',
        've',
        'value',
        'various',
        'very',
        'via',
        'viz',
        'vs',
        'want',
        'wants',
        'was',
        'wasn',
        'way',
        'we',
        'welcome',
        'well',
        'went',
        'were',
        'what',
        'whatever',
        'when',
        'whence',
        'whenever',
        'where',
        'whereafter',
        'whereas',
        'whereby',
        'wherein',
        'whereupon',
        'wherever',
        'whether',
        'which',
        'while',
        'whither',
        'who',
        'whoever',
        'whole',
        'whom',
        'whose',
        'why',
        'will',
        'willing',
        'wish',
        'with',
        'within',
        'without',
        'wonder',
        'would',
        'wouldn',
        'www',
        'yes',
        'yet',
        'you',
        'your',
        'yours',
        'yourself',
        'yourselves',
        'zero'
        ])
