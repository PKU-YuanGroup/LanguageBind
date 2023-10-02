import json
import os.path
from tqdm import tqdm

with open(r"G:\audioset\audioset\zip_audios\16k\eval.json", 'r') as f:
    data = json.load(f)['data']

new_data = []
total = 0
success = 0
for i in tqdm(data):
    total += 1
    video_id = os.path.basename(i['wav'])
    new_video_id = 'Y' + video_id
    i['wav'] = new_video_id
    if os.path.exists(f"G:/audioset/audioset/zip_audios/eval_segments/{i['wav']}") and not video_id.startswith('mW3S0u8bj58'):
        new_data.append(i)
        success += 1
print(total, success, total-success)
with open(r"G:\audioset\audioset\zip_audios\16k\filter_eval.json", 'w') as f:
    data = json.dump({'data': new_data}, f, indent=2)