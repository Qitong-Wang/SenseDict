import os
import json

import argparse

def print_funtion1(folder_path):

    #folder_path = './resultfullmcl/McGill-NLP__LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/no_revision_available/'

    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    # Sort the files by their modification time (earliest first)
    files_sorted = sorted(files)


    for file in files_sorted:
        if file.endswith('.json'):
            if "model_meta" in file:
                continue
            # Open and load the JSON file
            with open(file, 'r') as json_file:
                data = json.load(json_file)
                last_slash_index  = last_slash_index = file.rfind('/')
                file_name = file[last_slash_index + 1:]
                file_name = file_name[:-5]
                scores =  data['scores']
                if "test" in scores.keys():
                    main_score_test = data['scores']['test'][0]['main_score']
                    print(f"{main_score_test*100:.2f} {file_name}")
                elif "test_2021" in scores.keys():
                    main_score_test = data['scores']['test_2021'][0]['main_score']
                    print(f"{main_score_test*100:.2f} {file_name}")

                else:
                    main_score_test = data['scores']['train'][0]['main_score']
                    print(f"{main_score_test*100:.2f} {file_name} Train")


def print_funtion2(parent_path):
    
    #parent_path = './resulttest/'

    folders = [f for f in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, f))]
    sorted_folders = sorted(folders)
 
    for folder in sorted_folders:
        print("---------------")
        print(folder)
        print("---------------")
        folder_path = parent_path+folder+'/McGill-NLP__LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/no_revision_available/'

        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

        # Sort the files by their modification time (earliest first)
        files_sorted = sorted(files, key=os.path.getmtime)


        for file in files_sorted:
            if file.endswith('.json'):
                if "model_meta" in file:
                    continue
                # Open and load the JSON file
                with open(file, 'r') as json_file:
                    data = json.load(json_file)
                    last_slash_index  = last_slash_index = file.rfind('/')
                    file_name = file[last_slash_index + 1:]
                    file_name = file_name[:-5]
                    scores =  data['scores']
                    if "test" in scores.keys():
                        main_score_test = data['scores']['test'][0]['main_score']
                        print(f"{main_score_test*100:.2f} {file_name}")
                    elif "test_2021" in scores.keys():
                        main_score_test = data['scores']['test_2021'][0]['main_score']
                        print(f"{main_score_test*100:.2f} {file_name}")

                    else:
                        main_score_test = data['scores']['train'][0]['main_score']
                        print(f"{main_score_test*100:.2f} {file_name} [Train]")




if __name__ == "__main__": 
    count_dict = dict()

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='./CLS_encoding_train/mcl.mclpkl', help="output pkl file")
    args = parser.parse_args()

    print_funtion2(args.folder_path)
