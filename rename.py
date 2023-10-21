import os

i = 0
for filename in os.listdir('digits'):
    os.rename(f'digits\{filename}', f'digits\digit{i}.png')
    i += 1
