from asyncio.streams import protocols
import os
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor

PROT_A = Path('/workspace/gfp_function_prediction/data/boltz_results_test_input/predictions/test_input/test_input_model_0.cif')

def process_prot(prot):
    
    prot_cif =Path(prot) / f'{prot.name}_model_0.cif'
    out_path_es = Path(out_path) / f'{prot.name}.csv'  

    cmd = [
        'python', 'main.py',
        '--protA', str(PROT_A),
        '--protB', str(prot_cif),
        '--out', str(out_path_es)
    ]
    subprocess.run(cmd, cwd='/workspace/gfp_function_prediction/PDAnalysis')



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--mut_path', type=str, default='/workspace/gfp_function_prediction/data/outputs/boltz_results_subsample/predictions')
    parser.add_argument('--out', type=str, default='/workspace/gfp_function_prediction/data')
    args = parser.parse_args()

    dir_path = Path(args.mut_path)
    out_path = Path(args.out)
    timestamp = f'{datetime.now().strftime("%m%d_%H")}_es'
    out_path = os.path.join(out_path, timestamp)

    os.makedirs(out_path, exist_ok=True)

    # get the .cif files: 
    paths = [dir  for dir in dir_path.iterdir()]

    with ThreadPoolExecutor(max_workers=4) as executor:  # Tune workers based on your cores
        list(tqdm(executor.map(process_prot, paths), total=len(paths)))


    # # generate ES for each prediction:
    # for prot in tqdm(paths, desc='Predicting the Effective Strain of P:'):

    #     out_path_es = os.path.join(out_path, f'{prot.name}.csv')

    #     prot_cif = prot / f'{prot.name}_model_0.cif'


    #     # Run main.py in the specified directory
    #     cmd = [
    #         'python', 'main.py',
    #         '--protA', PROT_A,
    #         '--protB', str(prot_cif),
    #         '--out', str(out_path_es)
    #     ]
    #     subprocess.run(cmd, cwd='/workspace/gfp_function_prediction/PDAnalysis')



