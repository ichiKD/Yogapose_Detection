import os
import subprocess
for x in range(1, 12):
    input_folder = r'C:\Users\shubh\Desktop\Rishikesh\DATA SET\\'+str(x)
    output_folder = r'C:\Users\shubh\Desktop\Rishikesh\DATASET_MKV\\'+str(x)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mov')]
    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        output_file = os.path.splitext(input_file)[0] + '.mkv'
        output_path = os.path.join(output_folder, output_file)
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'ffv1',      # FFV1 video codec
            '-an',               # No audio
            output_path
        ]
        subprocess.run(cmd)
        