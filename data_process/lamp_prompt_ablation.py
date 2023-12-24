import re
import math

def contact(profile_entries, join_words = ", and "):
        
        return join_words.join(item for item in profile_entries)
    
def add_double_quote(text):
    return "\""+text+"\""

class LaMP2PromptAblation():
    def __init__(self) -> None:
        pass

    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        
        if len(retrieved_profile) == 0:
            return input
        
        max_profile_length = max_input_len-max_task_len
        profile_prompt = self.__per_profile_entity_prompt(
                                    retrieved_profile, 
                                    tokenizer, 
                                    max_profile_length)
    

        final_input_text = self.__merge_profile_and_input(input, profile_prompt)
        return final_input_text

    def __per_profile_entity_prompt(self, profile, tokenizer, max_profile_length):
        
        profile_entities = []
        
        final_profile = 'the category for the previous articles are'
        for item in profile:
            final_profile += add_double_quote(item['category'])
        
        
        original_profile_tokens = tokenizer.encode(final_profile)
        if len(original_profile_tokens) > max_profile_length:
            trim_length = math.ceil(len(original_profile_tokens)-max_profile_length)
            trim_text_tokens = original_profile_tokens[:-trim_length]
            trim_profile = tokenizer.decode(trim_text_tokens[:-1])
        else:
            trim_profile = final_profile
        
        trim_final_profile = trim_profile
        #trim_profile_text = tokenizer.decode(trim_profile_tokens[:-1])
        
        return trim_final_profile

    def __merge_profile_and_input(self, original_string, additional_string):

        return additional_string + '. ' + original_string


class LaMP3PromptAblation():
    def __init__(self) -> None:
        pass
    
    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        """
        To merge the task input and the user profile into the modified input
        """
        if len(retrieved_profile) == 0:
            return input
        
        input_length = len(tokenizer(input)['input_ids'])
        if input_length < 256:
            max_profile_length = (max_input_len-max_task_len)/len(retrieved_profile)
        else:
            max_profile_length = 0 if input_length>max_input_len else (max_input_len-input_length)/len(retrieved_profile)
        
        profile_prompt = self.__per_profile_entity_prompt(
                retrieved_profile, 
                tokenizer, 
                max_profile_length)
        

        final_input_text = self.__merge_profile_and_input(input, profile_prompt)
        return final_input_text

    def __per_profile_entity_prompt(self, profile, tokenizer, max_profile_length):
        
        profile_entities = []
        final_profile_template = ' is the score for the previous review'
        scores = [item['score'] for item in profile] 
        final_profile = ", ".join(scores)

        original_profile_tokens = tokenizer.encode(final_profile)
        if len(original_profile_tokens)+8 > max_profile_length:
            trim_length = math.ceil(len(original_profile_tokens)-max_profile_length+8)
            trim_text_tokens = original_profile_tokens[:-trim_length]
            trim_profile = tokenizer.decode(trim_text_tokens[:-1])
        else:
            trim_profile = final_profile
        
        trim_final_profile = trim_profile + final_profile_template
        #trim_profile_text = tokenizer.decode(trim_profile_tokens[:-1])
        # profile_entities.append(trim_final_profile)
        
        return trim_final_profile

    def __merge_profile_and_input(self, original_string, additional_string):

        return additional_string + '. ' + original_string
    

class LaMP4PromptAblation():
    def __init__(self) -> None:
        pass
    
    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        """
        To merge the task input and the user profile into the modified input
        """
        if len(retrieved_profile) == 0:
            return input
        
        max_profile_length = (max_input_len-max_task_len-5)
        profile_prompt = self.__per_profile_entity_prompt(
                retrieved_profile, 
                tokenizer, 
                max_profile_length)
        

        final_input_text = self.__merge_profile_and_input(input, profile_prompt)
        return final_input_text

    def __per_profile_entity_prompt(self, profile, tokenizer, max_profile_length):
        
        # profile_entities = []
        final_profile_template = ' is the title for previous articles'
        final_profile = ''
        for item in profile:
            final_profile += add_double_quote(item['title'])
        
        original_profile_tokens = tokenizer.encode(final_profile)
        if len(original_profile_tokens)+6 > max_profile_length:
            trim_length = math.ceil(len(original_profile_tokens)-max_profile_length+6)
            trim_text_tokens = original_profile_tokens[:-trim_length]
            trim_profile = tokenizer.decode(trim_text_tokens[:-1])
        else:
            trim_profile = final_profile
        
        trim_final_profile = trim_profile + final_profile_template
        #trim_profile_text = tokenizer.decode(trim_profile_tokens[:-1])
        # profile_entities.append(trim_final_profile)
        
        return trim_final_profile

    def __merge_profile_and_input(self, original_string, additional_string):

        return additional_string + '. ' + original_string
    

class LaMP5PromptAblation():
    def __init__(self) -> None:
        pass
    
    def aggregated_prompt(self, input, retrieved_profile, tokenizer, max_input_len=512, max_task_len=256):
        """
        To merge the task input and the user profile into the modified input
        """
        if len(retrieved_profile) == 0:
            return input
        
        max_profile_length = (max_input_len-max_task_len)
        profile_prompt = self.__per_profile_entity_prompt(
                retrieved_profile, 
                tokenizer, 
                max_profile_length)
        

        final_input_text = self.__merge_profile_and_input(input, profile_prompt)
        return final_input_text

    def __per_profile_entity_prompt(self, profile, tokenizer, max_profile_length):
        
        # profile_entities = []
        final_profile_template = ' is the title for the previous papers'
        final_profile = ''
        for item in profile:
            final_profile += add_double_quote(item['title'])

        original_profile_tokens = tokenizer.encode(final_profile)
        if len(original_profile_tokens)+7 > max_profile_length:
            trim_length = math.ceil(len(original_profile_tokens)-max_profile_length+7)
            trim_text_tokens = original_profile_tokens[:-trim_length]
            trim_profile = tokenizer.decode(trim_text_tokens[:-1])
        else:
            trim_profile = final_profile
        
        trim_final_profile = trim_profile + final_profile_template
        #trim_profile_text = tokenizer.decode(trim_profile_tokens[:-1])
        # profile_entities.append(trim_final_profile)
        
        return trim_final_profile

    def __merge_profile_and_input(self, original_string, additional_string):

        return additional_string + '. Following the given patterns ' + original_string

