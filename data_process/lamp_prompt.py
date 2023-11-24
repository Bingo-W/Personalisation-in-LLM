import re

def contact(profile_entries, join_words = ", and "):
        
        return join_words.join(item for item in profile_entries)
    
def add_double_quote(text):
    return "\""+text+"\""

class LaMP1Prompt():
    def __init__(self) -> None:
        pass

    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        
        max_profile_length = (max_input_len-max_task_len)/len(retrieved_profile)
        profile_prompt = contact(
            self.__per_profile_entity_prompt(
                retrieved_profile, 
                tokenizer, 
                max_profile_length)
        )
        
        #! I am not sure about the actual modification for the task input
        final_input_text = self.__merge_profile_and_input(input, profile_prompt)
        return final_input_text

    def __per_profile_entity_prompt(self, profile, tokenizer, max_profile_length):
        
        profile_entities = []
        for item in profile:
            
            trim_profile_tokens = tokenizer.encode(add_double_quote(item['title']))[:int(max_profile_length)]
            trim_profile_text = tokenizer.decode(trim_profile_tokens[:-1])
            profile_entities.append(trim_profile_text)

        return profile_entities
    
    def __merge_profile_and_input(self, original_string, additional_string):
        """
        To add the additional string into the first quote part in the original string
        param: orginal_string (str): the orginal sentence with some quoted part
        param: additional_string (str): the additional part will be added into the orginal part
        return: modified_string (str): the modified sentence
        """
        pattern = r'title "([^"]*)"'

        def replace(match):
            # This function is called for each match, and it replaces the match with the modified content
            quoted_text = match.group(1)  # Extract the text inside the quotes
            modified_text = f'"{quoted_text}" {additional_string}'  # Add the additional string
            return f'title {modified_text}'  # Reconstruct the entire quoted string

        # Use re.sub() to replace the quoted part with the modified content
        modified_string = re.sub(pattern, replace, original_string)

        return modified_string
    

class LaMP2Prompt():
    def __init__(self) -> None:
        pass

    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        
        max_profile_length = (max_input_len-max_task_len)/len(retrieved_profile)
        profile_prompt = contact(
            self.__per_profile_entity_prompt(
                retrieved_profile, 
                tokenizer, 
                max_profile_length)
        )

        final_input_text = self.__merge_profile_and_input(input, profile_prompt)
        return final_input_text

    def __per_profile_entity_prompt(self, profile, tokenizer, max_profile_length):
        
        profile_entities = []
        for item in profile:
            final_profile = 'the category for the article:' + add_double_quote(item['text']) + ' is ' + add_double_quote(item['category'])
            trim_profile_tokens = tokenizer.encode(final_profile)[:int(max_profile_length)]
            trim_profile_text = tokenizer.decode(trim_profile_tokens[:-1])
            profile_entities.append(trim_profile_text)
        
        return profile_entities

    def __merge_profile_and_input(self, original_string, additional_string):

        return additional_string + '. ' + original_string