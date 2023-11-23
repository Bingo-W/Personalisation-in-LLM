import re

class LaMP1Prompt():
    def __init__(self) -> None:
        pass

    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        

        max_profile_length = (max_input_len-max_task_len)/len(retrieved_profile)
        profile_prompt = self.__contact(
            self.__per_profile_entry_prompt(
                retrieved_profile, 
                tokenizer, 
                max_profile_length)
        )
        
        # 12341234
        #* adasd
        #! I am not sure about the actual modification for the task input
        final_input_text = self.__add_string_to_quote(input, profile_prompt)
        return final_input_text

    def __per_profile_entry_prompt(self, profile, tokenizer, max_profile_length):
        
        profile_entries = []
        for item in profile:
            
            trim_profile_tokens = tokenizer.encode(item['title'])[:int(max_profile_length)]
            trim_profile_text = tokenizer.decode(trim_profile_tokens)
            profile_entries.append(self.__add_double_quote(trim_profile_text))

        return profile_entries

    def __contact(self, profile_entries, join_words = ", and "):
        
        return join_words.join(item for item in profile_entries)


    def __add_double_quote(self, text):
        return "\""+text+"\""
    
    def __add_string_to_quote(self, original_string, additional_string):
        """
        To add the additional string into the first quote part in the original string
        param: orginal_string (str): the orginal sentence with some quoted part
        param: additional_string (str): the additional part will be added into the orginal part
        return: modified_string (str): the modified sentence
        """
        pattern = r'"([^"]*)"'

        def replace(match):
            # This function is called for each match, and it replaces the match with the modified content
            quoted_text = match.group(1)  # Extract the text inside the quotes
            modified_text = f'{quoted_text} {additional_string}'  # Add the additional string
            return f'"{modified_text}"'  # Reconstruct the entire quoted string

        # Use re.sub() to replace the quoted part with the modified content
        modified_string = re.sub(pattern, replace, original_string)

        return modified_string