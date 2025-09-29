import os, io, sys
import muspy
from pathlib import Path
import miditoolkit
import json

class MuteWarn:
  def __enter__(self):
    self._init_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._init_stdout


def save_wav_from_midi(midi_fn, file_name, qpm=80):
  if isinstance(midi_fn, Path):
    midi_fn = str(midi_fn)
  assert isinstance(midi_fn, str)
  with MuteWarn():
    music = muspy.read_midi(midi_fn)
    music.tempos[0].qpm = qpm
    music.write_audio(file_name, rate=16000)

def get_median_tempo(tempo_changes):
  tempos = []
  for tempo in tempo_changes:
      tempos.append(tempo.tempo)
  if len(tempos) == 0:
    tempo = 120 # default
  else:
    tempo = sorted(tempos)[int(len(tempos)/2)]
  return tempo

def save_wav_from_performance_midi(in_fp, out_fp):
  midi_obj = miditoolkit.midi.parser.MidiFile(in_fp)
  mid_tempo = get_median_tempo(midi_obj.tempo_changes)
  save_wav_from_midi(in_fp, out_fp, qpm=mid_tempo)

if __name__ == '__main__':   
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', help='directory of midi files')
  args = parser.parse_args()
  in_dir = Path(args.dir)
  # out_dir = Path('../post-ai-samples/output_midi/*/')
  # out_dir = Path(f'../pop1k7/segment_first_{int(n)}sec/')

  # out_dir.mkdir(parents=True, exist_ok=True)
  # remove all wav files
  # remove_wav_files = list(in_dir.glob('*.wav'))
  # for remove_wav_file in remove_wav_files:
  #   remove_wav_file.unlink()

  in_files = sorted(list(in_dir.glob('*.midi')))   
  # print(in_files)
  
  # calculate the middle value of the tempo
  for in_fp in in_files:
      midi_obj = miditoolkit.midi.parser.MidiFile(in_fp)
      tempo = get_median_tempo(midi_obj.tempo_changes)
      save_wav_from_midi(in_fp.parent / in_fp.name, in_fp.parent / (in_fp.stem +'_rendered.wav'), qpm=tempo)