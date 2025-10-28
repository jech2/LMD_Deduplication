import json
import numpy as np
from pathlib import Path

from miditok import TokenizerConfig
from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.tokenizations import Octuple
from miditok.utils import get_midi_ticks_per_beat
from symusic import Score
from symusic.core import ScoreTick


class CustomOctuple(Octuple):
    def __init__(self,
        tokenizer_config=None,
        params=None):
        if tokenizer_config == None:
            tokenizer_config = self.get_reduced_octuple_config()
        
        super().__init__(tokenizer_config, params)
        # self.durations = self._create_durations_tuples()
        self.__create_vocabulary()
        # self.one_token_stream = False
    
    def get_reduced_octuple_config(self):
        config = TokenizerConfig(num_velocities=32, 
                            use_pitchdrum_tokens=True,
                            pitch_range=(0, 128), # inclusive
                            drums_pitch_range=(0, 128), # inclusive
                            use_programs=True, 
                            programs=list(range(-1, 128)),
                            max_bar_embedding=256,
                            bar_resolution=64, 
                            )
        return config
    
    def __create_vocabulary(self) -> None:
        r"""
        Create the vocabulary of the tokenizer as a dictionary.

        This method is called during the tokenizer's initialization, and requires
        ``_create_vocabulary`` to be implemented by a child class.
        """
        vocab = self._create_base_vocabulary()

        if isinstance(vocab[0], list):  # multi-voc
            self._vocab_base = [{} for _ in range(len(vocab))]
            self.__vocab_base_inv = [{} for _ in range(len(vocab))]
            for vid in range(len(vocab)):
                vocab[vid] = self.special_tokens + vocab[vid]
                for tok in vocab[vid]:
                    self.add_to_vocab(tok, vid)
        else:
            vocab = self.special_tokens + vocab
            for tok in vocab:
                self.add_to_vocab(tok)
    
    def _create_base_vocabulary(self):
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MIDITokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = [[] for _ in range(5)]

        # PITCH
        vocab[0] += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        if self.config.use_pitchdrum_tokens:
            vocab[0] += [
                f"PitchDrum_{i}" for i in range(*self.config.drums_pitch_range)
            ]

        # VELOCITY
        vocab[1] += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab[2] += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # POSITION
        # self.time_division is equal to the maximum possible ticks/beat value.
        num_positions = self.config.additional_params['bar_resolution'] * 2
        vocab[3] += [f"Position_{i}" for i in range(num_positions)]

        # BAR (positional encoding)
        vocab[4] += [
            f"Bar_{i}"
            for i in range(self.config.additional_params["max_bar_embedding"])
        ]

        # PROGRAM
        if self.config.use_programs:
            vocab.append([f"Program_{i}" for i in self.config.programs])

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # TIME_SIGNATURE
        if self.config.use_time_signatures:
            vocab.append([f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures])

        return vocab
    
    def _midi_to_tokens(self, midi):
        r"""
        Convert a **preprocessed** MIDI object to a sequence of tokens.

        The workflow of this method is as follows: the global events (*Tempo*,
        *TimeSignature*...) and track events (*Pitch*, *Velocity*, *Pedal*...) are
        gathered into a list, then the time events are added. If `one_token_stream` is
        ``True``, all events of all tracks are treated all at once, otherwise the
        events of each track are treated independently.

        :param midi: the MIDI :class:`symusic.Score` object to convert.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        # Create events list
        all_events = []
        if not self.one_token_stream:
            if len(midi.tracks) == 0:
                all_events.append([])
            else:
                all_events = [[] for _ in range(len(midi.tracks))]

        # Global events (Tempo, TimeSignature)
        global_events = self._create_midi_events(midi)
        if self.one_token_stream:
            all_events += global_events
        else:
            for i in range(len(all_events)):
                all_events[i] += global_events

        # Compute ticks_per_beat sections depending on the time signatures
        # This has to be computed several times, in preprocess after resampling & here.
        if (
            not self._note_on_off
            or (self.config.use_sustain_pedals and self.config.sustain_pedal_duration)
            or self.config.use_chords
            or self.config.use_pitch_intervals
        ):
            if self.config.use_time_signatures:
                ticks_per_beat = get_midi_ticks_per_beat(midi)
            else:
                ticks_per_beat = np.array([[midi.end(), self.time_division]])
        else:
            ticks_per_beat = None

        # Adds track tokens
        for ti, track in enumerate(midi.tracks):
            track_events = self._create_track_events(track, ticks_per_beat)
            if self.one_token_stream:
                all_events += track_events
            else:
                if self.config.program_changes:
                    # ProgramNoteOff desc to make sure it appears before Pedals and
                    # everything else
                    track_events.insert(
                        0, Event("Program", track.program, 0, desc="ProgramNoteOff")
                    )
                all_events[ti] += track_events
                self._sort_events(all_events[ti])
        if self.one_token_stream:
            self._sort_events(all_events)
            # Add ProgramChange (named Program) tokens if requested.
            if self.config.program_changes:
                self._insert_program_change_events(all_events)

        # Add time events
        if self.one_token_stream:
            all_events = self._add_time_events(all_events, midi.ticks_per_quarter)
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                all_events[i] = self._add_time_events(
                    all_events[i], midi.ticks_per_quarter
                )
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

    def save_tokens_str(
        self,
        tokens,
        path,
        programs=None,
        **kwargs,
    ) -> None:
        r"""
        Save tokens as a JSON file.

        In order to reduce disk space usage, **only the ids are saved**. Use ``kwargs``
        to save any additional information within the JSON file.

        :param tokens: tokens, as list, numpy array, torch or tensorflow Tensor.
        :param path: path of the file to save.
        :param programs: (optional), programs of the associated tokens, should be given
            as a tuples (int, bool) for (program, is_drum).
        :param kwargs: any additional information to save within the JSON file.
        """
        tokens_str = self._ids_to_tokens(tokens)

        with Path(path).open("w") as outfile:
            dic = {"events": tokens_str, **kwargs}
            if programs is not None:
                dic["programs"] = programs
            json.dump(dic, outfile)
            
    @staticmethod
    def _events_to_tokens(
        events,
        ):
        r"""
        Convert a sequence of ``Events`` to their tokens format (str).

        :param events: sequence of Events to convert.
        :return: the sequence of corresponding tokens (str).
        """
        tokens = []
        if len(events) == 0:
            return tokens
        if isinstance(events[0], list):  # multiple vocabularies
            # cannot use recursion here because of the vocabulary type id
            ret = [[str(event) for event in multi_event] for multi_event in events]
            ret = [multi_event_strs for multi_event_strs in ret if int(multi_event_strs[4].split('_')[1]) < 256]
            return ret
        return [str(event) for event in events]