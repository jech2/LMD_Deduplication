from miditok import REMI, Octuple, TokenizerConfig
from pathlib import Path
from symusic import Score
import torch
import numpy as np
import random
import copy

class Augmentator:
    def __init__(self, vocab):
        self.vocab = vocab
        
        self.min_pitch = 0
        self.max_pitch = 127
        self.min_vel = 0
        self.max_vel = 31
        self.min_tempo = 0 
        self.max_tempo = 48

        self.aug_pitch = range(-6, 7)
        self.aug_vel = range(-3, 4)
        self.aug_tempo = range(-2, 2)

        self.min_bar = 0
        self.max_bar = 255
        self.aug_bar = range(1, 5) # max 4 bar shift
        
        self.drop_note_prob = 0.15
        self.mask_idx = self.vocab.special_tokens.index('MASK_None') 

        self.use_reduced_vocab = True if len(self.vocab._vocab_base) != 8 else False

    # get the name and index of octuple each channel type
    def change_instrument_order(self, events):
        events_midi = self.vocab(events[1:-1])
        tracks = list(events_midi.tracks)
        random.shuffle(tracks)
        events_midi.tracks = tracks
        try:
            events_shuffled = self.vocab(events_midi)
            events[1:-1] = torch.tensor(events_shuffled)
        except Exception as e:
            print('event shuffle failed', e)
        return events
        
    def drop_instruments(self, events):
        if isinstance(events, np.ndarray):
            events = torch.from_numpy(events)
        all_played_instruments = torch.unique(events[1:-1, self.vocab.vocab_types_idx['Program']])
        # drum_idx = (all_played_instruments == len(self.vocab.special_tokens)).nonzero(as_tuple=True)[0]
        # if len(drum_idx) > 0:
            # all_played_instruments = torch.cat((all_played_instruments[:drum_idx], all_played_instruments[drum_idx + 1:]))
        
        if all_played_instruments.numel() <= 2:
            return events
        
        random_inst_drop = torch.rand(all_played_instruments.size(0)) < 0.5
        while torch.sum(random_inst_drop) > all_played_instruments.size(0) // 2:
            random_inst_drop = torch.rand(all_played_instruments.size(0)) < 0.5
        
        instruments_to_drop = all_played_instruments[random_inst_drop]
        mask = ~(events[:, self.vocab.vocab_types_idx['Program']].unsqueeze(-1) == instruments_to_drop).any(-1)
        filtered_events = events[mask]
        return filtered_events
    
    
    def check_extreme(self, events, type, low, high):
        for i in range(len(events)):
            type_idx = self.vocab.vocab_types_idx[type]
            inst_idx = self.vocab.vocab_types_idx['Program']
            offset = len(self.vocab.special_tokens)
            value = events[i][type_idx] - offset

            # skip cases where the value is special tokens
            if events[i][type_idx] < offset:
                continue

            # # skip drum for pitch augmentation
            if type == 'Pitch' and events[i][inst_idx] == offset:
                continue
            
            assert type in ['Pitch', 'Velocity', 'Tempo', 'Bar'] , 'type should be either Pitch or Velocity or Tempo or Bar (int conversion of value)' 
            low = min(low, value)
            high = max(high, value)

        return low, high

    def augment_pitch_octave(self, events):
        # for each track, randomly shift the pitch by 12, 24 or -12, -24
        all_tracks = torch.unique(events[1:-1, self.vocab.vocab_types_idx['Program']])
        for track in all_tracks:
            if track == len(self.vocab.special_tokens) or track > 113: # skip drum track
                continue
            track_idx = (events[:, self.vocab.vocab_types_idx['Program']] == track).nonzero(as_tuple=True)[0]
            random_shift = random.choice([12, 24, -12, -24])
            # if the vocab range exceeds the limit, then skip the augmentation
            if (events[track_idx, self.vocab.vocab_types_idx['Pitch']] + random_shift).min() < self.min_pitch or (events[track_idx, self.vocab.vocab_types_idx['Pitch']] + random_shift).max() > self.max_pitch:
                continue 
            else:
                events[track_idx, self.vocab.vocab_types_idx['Pitch']] += random_shift        
        return events
    
    # the unit is 1/32
    def shift_onset(self, events, p=0.3):
        # for each note, randomly shift the onset by 1, 2, -1, -2 except the drum
        offset = len(self.vocab.special_tokens)
        # first, generate masks
        assert events.size(0) > 2, 'event size is too small'
        mask = torch.rand(events.size(0)-2) < p
        # mask = mask & ~(events[1:-1, self.vocab.vocab_types_idx['Program']] == len(self.vocab.special_tokens)) # skip drum track
        # random sample from -1, 1
        random_shift = torch.randint(-2, 3, (events.size(0)-2,)) * mask
        # apply the mask
        events[1:-1, self.vocab.vocab_types_idx['Position']] += random_shift
        # clip the position value to 0, 127
        events[1:-1, self.vocab.vocab_types_idx['Position']] = torch.clamp(events[1:-1, self.vocab.vocab_types_idx['Position']], offset, 127+offset)
        
        return events
        
    def shift_duration(self, events, p=0.7):
        # for each note, randomly shift the duration by 1, 2, -1, -2 except the drum
        offset = len(self.vocab.special_tokens)
        # first, generate masks
        mask = torch.rand(events.size(0)-2) < p
        # mask = mask & ~(events[1:-1, self.vocab.vocab_types_idx['Program']] == len(self.vocab.special_tokens)) # skip drum track
        # random sample from -1, 1
        random_shift = torch.randint(-4, 5, (events.size(0)-2,)) * mask
        # apply the mask
        events[1:-1, self.vocab.vocab_types_idx['Duration']] += random_shift # make sure that the duration is not negative
        # clip the position value to 0, 127
        events[1:-1, self.vocab.vocab_types_idx['Duration']] = torch.clamp(events[1:-1, self.vocab.vocab_types_idx['Duration']], offset, 63+offset)
        
        return events

    def random_bar_drop(self, events, p=0.15):
        # for each track, randomly drop some bar
        all_tracks = torch.unique(events[1:-1, self.vocab.vocab_types_idx['Program']])
        offset = len(self.vocab.special_tokens)
        total_mask = torch.zeros(events.size(0)-2)
        for track in all_tracks: 
            # first, generate masks of bars
            total_bars = max(events[1:-1, self.vocab.vocab_types_idx['Bar']]) - offset
            bar_mask = torch.rand(total_bars) < p # get random mask of bars
            # mask events when bar_mask[bar] is true
            mask = torch.zeros(events.size(0)-2).bool()
            for bar in range(total_bars):
                if bar_mask[bar]:
                    bar_mask_track = events[1:-1, self.vocab.vocab_types_idx['Bar']] == bar + offset
                    mask = mask + bar_mask_track
            
            mask = mask & (events[1:-1, self.vocab.vocab_types_idx['Program']] == track) # get current track mask
            total_mask = total_mask + mask

        mask = ~total_mask.bool()
        filtered_events = events[1:-1][mask]
        return filtered_events

    def augment_fixed_random_val(self, events, type, min_limit, max_limit, aug_range):
        assert type in ['Pitch', 'Velocity', 'Tempo', 'Bar'] , 'type should be either Pitch or Velocity or Tempo or Bar (int conversion of value)' 

        min_val, max_val = self.check_extreme(events, type=type, low=max_limit, high=min_limit)
        if min_val == min_limit and max_val == max_limit:
            return events

        num_key = random.choice(aug_range)
        while min_val + num_key < min_limit or max_val + num_key > max_limit:
            num_key = random.choice(aug_range)
        

        for i in range(1, len(events)-1): # skip bos and eos tokens
            type_idx = self.vocab.vocab_types_idx[type]
            inst_idx = self.vocab.vocab_types_idx['Program']
            offset = len(self.vocab.special_tokens) # drum: Pitch_-1 = offset
            
            # skip drum for pitch augmentation
            if type == 'Pitch' and events[i][inst_idx] == offset:
                continue

            events[i][type_idx] += num_key
        return events

    def augment_random_val(self, events, type, min_limit, max_limit, aug_range=None):
        assert type == 'Velocity' or type=='Duration' or type=='Bar', 'type should be either Velocity or Duration or Bar (int conversion of value)' 

        min_val, max_val = self.check_extreme(events, type=type, low=max_limit, high=min_limit)
        upper_limit = max_limit - max_val
        lower_limit = min_limit - min_val
        if aug_range is None:
            # random seq gen 
            num_key = np.random.randint(lower_limit, upper_limit, size=len(events))
        else:
            assert max(lower_limit, aug_range[0]) <= min(upper_limit, aug_range[-1]), f'aug_range should be within the limit: current aug_range: {max(lower_limit, aug_range[0])}, {min(upper_limit, aug_range[-1])}'
            upper_limit = min(upper_limit, aug_range[-1])
            lower_limit = max(lower_limit, aug_range[0])
            if lower_limit == 0 and upper_limit == 0:
                return events
            else:
                num_key = np.random.randint(lower_limit, upper_limit, size=len(events))

        for i in range(1, len(events)-1): # skip bos and eos tokens
            type_idx = self.vocab.vocab_types_idx[type]
            inst_idx = self.vocab.vocab_types_idx['Program']
            offset = len(self.vocab.special_tokens) # drum: Pitch_-1 = offset
            
            # skip drum for pitch augmentation
            if type == 'Pitch' and events[i][inst_idx] == offset:
                continue

            events[i][type_idx] += num_key[i]
        return events

    def change_instrument(self, events):
        inst_idx = self.vocab.vocab_types_idx['Program']
        offset = len(self.vocab.special_tokens)
        
        # get unique number of instruments
        all_played_instruments = [int(events[i, inst_idx]) for i in range(len(events))]
        all_played_instruments = set(all_played_instruments)
        # remove some instrument numbers
        exclude_midi_indices = list(range(0, offset + 1)) + list(range(113 + offset, 129 + offset)) # exclude special tokens + Percussive + Sound Effects 
        all_played_instruments = set(all_played_instruments) - set(exclude_midi_indices) # the list of instruments to change
        
        possible_inst_values = list(range(0, len(self.vocab._vocab_base[inst_idx]))) # possible values for instrument vocab
        possible_inst_values = list(set(possible_inst_values) - set(exclude_midi_indices)) # the list of instrument for changing
        
        chged_inst = np.random.choice(possible_inst_values, len(all_played_instruments))
        change_mapping = {k: v for k, v in zip(all_played_instruments, chged_inst)}
        
        for i in range(len(events)):
            if int(events[i][inst_idx]) in change_mapping.keys():
                events[i][inst_idx] = change_mapping[int(events[i][inst_idx])]
                
        return events
    
        
    def drop_note_augment(self, events):    
        # print('drop_note_augment')
        n_tokens = [len(self.vocab._vocab_base[i]) for i in range(len(self.vocab._vocab_base))]
        n_types = len(self.vocab._vocab_base)
        for i in range(len(n_tokens)):
            assert max(events[:, i]) < n_tokens[i]
        
        # make the sequence as 1-dimensional array
        item = events.flatten()
        # print(item.shape)
        sz = len(item)
        assert (
            self.mask_idx not in item
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            self.mask_idx,
        )
        # assert not self.mask_whole_words, 'mask whole words not supported for cp'

        def generate_mask(sz, prob):
            mask_n = np.random.rand(sz)
            mask_s = np.zeros(sz, dtype=np.int8)
            mask_s += mask_n < prob
            return mask_s
        mask_prob = self.drop_note_prob
        mask = np.zeros_like(item, dtype=np.int8)
        mask[n_types:-n_types] = np.repeat(generate_mask(sz // n_types - 2, mask_prob), n_types) # consider bos and eos tokens

        # calculate random mask for note to drop
        dropped_item = item[mask != 1]
        dropped_events = dropped_item.reshape(-1, len(self.vocab._vocab_base))

        return dropped_events   
    
    def adjust_bar_shift(self, events):
        bar_shift = 0
        bar_type_idx = self.vocab.vocab_types_idx["Bar"]
        for i in range(len(events)):
            bar_token_val = events[i][bar_type_idx]
            if bar_token_val < len(self.vocab.special_tokens):
                continue
            else:
                bar_shift = (bar_token_val - len(self.vocab.special_tokens)) # the first bar value in the current segment
                assert bar_shift >= 0 and bar_shift < len(self.vocab._vocab_base[bar_type_idx]), f"bar_shift: {bar_shift} is out of range"
                break
        
        for i, event in enumerate(events):
            if event[bar_type_idx] >= len(self.vocab.special_tokens):
                events[i][bar_type_idx] -= bar_shift     
        return events
    
    def __call__(self, events):
        title = ['sample']
        
        # change instrument order
        if random.random() > 0.5:
            title.append('change_instrument_order')
            events = self.change_instrument_order(events)

        # drop note augmentation
        if random.random() > 0.5:
            title.append('drop_instrument')
            events = self.drop_instruments(events)
            
        if random.random() > 0.5:
            title.append('drop_note_augment')            
            events = self.drop_note_augment(events)
            
        # random bar drop augmentation
        if random.random() > 0.5:
            title.append('random_bar_drop')
            events = self.random_bar_drop(events)
        
        # pitch augmentation
        if random.random() > 0.5:
            title.append('pitch_augment_fixed_random_val')
            events = self.augment_fixed_random_val(
                events,
                type="Pitch",
                min_limit=self.min_pitch,
                max_limit=self.max_pitch,
                aug_range=self.aug_pitch,
            )
            
        # pitch octave augmentation
        if random.random() > 0.5:
            title.append('pitch_octave_augment')
            events = self.augment_pitch_octave(events)
        
        # onset augmentation
        if random.random() > 0.5:
            title.append('onset_augment')
            events = self.shift_onset(events)
        
        # duration augmentation
        if random.random() > 0.5:
            title.append('duration_augment')
            events = self.shift_duration(events)
        
        # velocity augmentation
        if random.random() > 0.5:
            title.append('velocity_augment_random_vel')
            events = self.augment_random_val(
                events,
                type="Velocity",
                min_limit=self.min_vel,
                max_limit=self.max_vel,
                aug_range=self.aug_vel,
            )
            
        if random.random() > 0.5:
            title.append('bar shift_fixed_random_val')
            events = self.augment_fixed_random_val(
                events,
                type="Bar",
                min_limit=self.min_bar,
                max_limit=self.max_bar,
                aug_range=self.aug_bar,
            )
            
        # tempo augmentation
        if not self.use_reduced_vocab:
            if random.random() > 0.5:
                title.append('tempo_augment')
                events = self.augment(
                    events,
                    type="Tempo",
                    min_limit=self.min_tempo,
                    max_limit=self.max_tempo,
                    aug_range=self.aug_tempo,
                )
    
        # change instrument
        if random.random() > 0.5:
            title.append('change_instrument')
            events = self.change_instrument(events)

        title = '_'.join(title)
        
        return events, title