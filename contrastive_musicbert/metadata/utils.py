import re
from pathlib import Path

def escape_special_chars(text):
    # escape special characters: () .
    return re.sub(r'([().])', r'\\\1', text)

def filepath_to_artist_and_title(filepath):
    # e.g.: "../data/clean_midi/ABBA/Dancing_Queen.mid" -> ("ABBA", "Dancing Queen")
    filepath = Path(filepath)
    artist = filepath.parent.name
    title = filepath.stem
    return artist, title

def filepath_to_artist_and_title_str(filepath):
    artist, title = filepath_to_artist_and_title(filepath)
    return f'{artist}__{title}'

def artist_and_title_str_to_filepath(artist_and_title_str, ref_dir, ref_ext='.mid'):
    artist, title = artist_and_title_str.split('__')
    if ref_dir[-1] == '/':
        ref_dir = ref_dir[:-1]
    return f'{ref_dir}/{artist}/{title}{ref_ext}'
