import os
import shutil
import re
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBart50Tokenizer
import google.protobuf
import sentencepiece as spm
import pandas as pd

def main():
    # The preprocess class can be initialized with different steps activated (depending on what you want)
    # You can off course do all the steps at once, but for testing it might be nice to do it in part
    preprocess = PreprocessFromMap(r'00 2\00', 'txt_00 2',
                                   to_map=False,
                                   edit_txt=False,
                                   divide_into_folders=False,
                                   translate=False,
                                   construct_df=True)



# The preprocessing is wrapped up in a class for a few reasons:
# It looks tidier; it can be reused in different scripts for this project with ease
# ;it makes adding more functions in the future easier
class PreprocessFromMap:
    def __init__(self, data_direct, target_direct, to_map=False,
                 edit_txt=False, divide_into_folders=False, translate=False, construct_df=False):
        # Initialization of variables
        self.data_direct = data_direct
        self.target_direct = target_direct
        self.to_map = to_map
        self.edit_txt = edit_txt
        self.translation_directory = f'{self.target_direct}_translation'

        # Specifying which function you want to call. For testing individually etc.
        if to_map is True:
            self.txt_extract()
        if edit_txt is True:
            self.txt_edit()

        # These two functions have a 'desired_length' parameter associated with it.
        # By default, this is set at thirty
        # If you want to translate bigger chunks of files, you can change this in both functions
        if divide_into_folders is True:
            self.divide_into_folders_translation()
        if translate is True:
            # For this parameter, input a list (can be: [0] if you just want to translate one folder)
            folder_number_to_translate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.translate(folder_to_translate=folder_number_to_translate)

        if construct_df is True:
            self.to_df()

    # Function to extract the .txt files and rename them
    def txt_extract(self):
        # Make the new folder
        if not os.path.exists(self.target_direct):
            os.makedirs(self.target_direct)

        # Loop through the original folder
        for video_folder in os.listdir(self.data_direct):
            item_path = os.path.join(self.data_direct, video_folder)

            if os.path.isdir(item_path):
                # Loop through the files in each directory
                for file in os.listdir(item_path):

                # Takes only the .txt files
                    if file.endswith('.txt'):
                        txt_file_path = os.path.join(item_path, file)

                        # Creates a new name for the .txt file
                        new_file = f'{file[0:3]}_{video_folder}.txt'
                        # Copies the file with the new name to the target directory
                        copy_destination = os.path.join(self.target_direct, new_file)
                        shutil.copy(txt_file_path, copy_destination)
    # Loops through the txt files and edits the text within
    # To remove timestamps and the 'WEBVTT' text
    def txt_edit(self):
        # Loops through each txt file
        for txt in os.listdir(self.target_direct):
            with open(os.path.join(self.target_direct, txt), 'r', encoding='utf=8') as file:
                txt_content = file.readlines()

            # After opening, this function edits each file
            new_text = self.trim_txt(txt_content)
            # Room for translation function?

            # It writes the new content to the file
            with open(os.path.join(self.target_direct, txt), 'w', encoding='utf-8') as file:
                file.writelines(new_text)

    def trim_txt(self, string):
        # Presents a base of the new text
        new_text = ''
        # Loops through each word, except the first one (the 'WEBVTT' handle)
        for word in string[1:]:
            # If it finds this pattern, it will skip the word in the loop
            if re.match(r"\b(\d{2}):(\d{2}):(\d{2})\.(\d{3})\b", word):
                continue

            # Removes additional non-informational text
            # Match cases is used to leave open the possibility for adding more handles in the future if needed
            match word:
                case 'WEBVTT':
                    continue
                case '-->':
                    continue
                case _:
                    new_text += word
        return new_text

    # Function to divide txt files into separate folder for translation.
    # This is done to translate the entire nld texts in parts.
    def divide_into_folders_translation(self, desired_amount=30):
        # Make a directory for it
        if not os.path.exists(self.translation_directory):
            os.mkdir(self.translation_directory)
        # Pick out all the Dutch files and copy them
        for txt in os.listdir(self.target_direct):
            if txt.startswith('nld'):
                source_file = os.path.join(self.target_direct, txt)
                destination_file = os.path.join(self.translation_directory, txt)
                shutil.copy(source_file, destination_file)

        part_now = 1
        # Store all the files in a list
        translation_dir = [file for file in os.listdir(self.translation_directory)]

        # Loop through the files and move them in chunks of 30 to new folders
        for i in range(0, len(translation_dir), desired_amount):
            # Create a new folder for each part
            part_folder = os.path.join(self.translation_directory, f'{self.target_direct}_{part_now}')

            # Create the folder if it doesn't exist
            if not os.path.exists(part_folder):
                os.mkdir(part_folder)

            # Select up to 30 files to move to this folder
            files_to_move = translation_dir[i:i + desired_amount]

            # Move all files to their designated folder
            for txt in files_to_move:
                source_file = os.path.join(self.translation_directory, txt)
                destination_folder = os.path.join(part_folder, txt)
                shutil.move(source_file, destination_folder)
            # Move on to the next part
            part_now += 1


    # Function to initialize the model. In this case: facebook/mbart
    def translation_model_init(self, text):
        # The model is loaded onto the GPU with cuda (requires Nvidia GPU).
        model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt').to('cuda')

        tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        tokenizer.src_lang = 'nl_XX'

        # Tokenize the text
        encoded = tokenizer(text, return_tensors='pt').to('cuda')
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=tokenizer.lang_code_to_id['en_XX']
        )
        # Translate the text from the tokens
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation

    def translate(self, folder_to_translate, desired_amount=30):
        # Stores all folders that need translating (so not the ones already translated).
        translate_dir = [folder for folder in os.listdir(self.translation_directory) if not folder.endswith('translated')]
        print(translate_dir)
        # folder_to_translate is a global list variable that indicates which folder you want it to translate
        for number in folder_to_translate:
            folder = translate_dir[number]
            folder_path = os.path.join(self.translation_directory, folder)
            print(folder_path)

            current_count = 0
            # For each file in the folder, read it, and translate it
            for file in os.listdir(folder_path):
                text_path = os.path.join(folder_path, file)
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Translation is done in part for some text files, since some were too big for the model
                translations = []
                # Uses different function for possible splitting of the text
                for part in self.split_text(content):
                    individual_part = self.translation_model_init(part)
                    translations.extend(individual_part)
                # After having translated each part in the list, it will join the list in a single string
                translation = ' '.join(translations)

                # Write the translation to the file and move onto the next
                with open(text_path, 'w', encoding='utf-8') as f:
                    for sentence in translation:
                        f.write(sentence)
                current_count += 1
                print(f'{current_count}/{desired_amount}')
            print(f'Process done - succesfully translated {desired_amount} files')
            # Move on to the next folder and rename the previous one
            new_folder_name = f'{folder}_translated'
            new_folder_name_path = os.path.join(self.translation_directory, new_folder_name)
            os.rename(folder_path, new_folder_name_path)

    # Function for splitting the larger text files
    # Max_length is chosen by finding a large file that could be translated all at once
    def split_text(self, text, max_length=750):
        # No splitting if the text is small enough.
        # This is done to keep context of the entire file and save computation
        if len(text) <= max_length:
            return [text]
        # Splits the entire text by the largest allowed enter (/n).
        parts = []
        while len(text) > max_length:
            split = text[:max_length].rfind(r'\n')
            if split == -1:
                split = max_length
            parts.append(text[:split + 1].strip())
            text = text[split +1:].strip()

        # appends the text to the parts
        if text:
            parts.append(text)

        return parts

    # Final function to put all english and Dutch (translated) files in one dataframe
    def to_df(self):
        # First takes all english files and their name
        eng_files_path = self.target_direct
        dataframe_eng = pd.DataFrame()
        eng_files = []
        eng_files_name = []
        for files in os.listdir(eng_files_path):
            if files.startswith('eng'):
                with open(os.path.join(eng_files_path, files), 'r', encoding='utf-8') as f:
                    content = f.read()
                content_stripped = content.replace('\n', ' ')
                eng_files.append(content_stripped)
                eng_files_name.append(files)

        # And puts them into a pandas df
        dataframe_eng['file_name'] = eng_files_name
        dataframe_eng['original_text'] = eng_files

        # Now take all the Dutch (translated) files
        nld_files_path = f'{self.target_direct}_translation'
        dataframe_nld = pd.DataFrame()
        nld_files = []
        nld_files_name = []
        for folders in os.listdir(nld_files_path):
            folder_path = os.path.join(nld_files_path, folders)
            # Only if it is translated
            if folder_path.endswith('translated'):
                for file in os.listdir(folder_path):
                    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as nld_file:
                        nld_content = nld_file.read()

                    nld_files.append(nld_content)
                    nld_files_name.append(file)

        # Adds them to another pandas df
        dataframe_nld['file_name'] = nld_files_name
        dataframe_nld['original_text'] = nld_files

        # Merges the rows of the Dutch (translated) df and the English df
        complete_df = pd.concat([dataframe_eng, dataframe_nld], axis=0, ignore_index=True)

        # Saves it to csv to be able to work with it in a different script
        # The script gets so large otherwise :)
        complete_df.to_csv('translated_df.csv', index=False)
        return complete_df


if __name__ == '__main__':
    main()