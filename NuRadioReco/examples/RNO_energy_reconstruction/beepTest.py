import os
duration = 0.25  # seconds
C = 262  # Hz
E = 327.5
G = 393


os.system(f'play -nq -t alsa synth {duration} sine {C}')
os.system(f'play -nq -t alsa synth {duration} sine {E}')
os.system(f'play -nq -t alsa synth {duration} sine {G}')

import time
time.sleep(1)

os.system(f'play -nq -t alsa synth {duration} sine {C}')
os.system(f'play -nq -t alsa synth {duration} sine {E}')
os.system(f'play -nq -t alsa synth {duration} sine {G}')
