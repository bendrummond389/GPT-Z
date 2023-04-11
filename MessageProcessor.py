import os
import json


class MessageProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_file = "processed_data.txt"

    def process_json_files(self):
        finetuning_data = []

        for filename in os.listdir(self.input_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(self.input_folder, filename)
                raw_messages = self.open_json_file(file_path)
                filtered_messages = self.remove_attributes(raw_messages["messages"])
                prompt_response_pairs_v2 = self.create_prompt_response_pairs_v2(
                    filtered_messages
                )
                finetuning_data += self.prepare_data_for_finetuning(
                    prompt_response_pairs_v2
                )

        self.clean_finetuning_data(finetuning_data, self.output_file)

    @staticmethod
    def open_json_file(file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data

    @staticmethod
    def remove_attributes(messages):
        new_messages = []

        for message in messages:
            content = message.get("content", "")
            new_message = {
                "sender_name": message.get("sender_name"),
                "timestamp_ms": message.get("timestamp_ms"),
                "content": content,
            }
            if (
                content != ""
                and "sent an attachment"
                and "changed the group photo" not in content
                and "to your message" not in content
                and "http" not in content
                and "https" not in content
                and "www" not in content
            ):
                new_messages.append(new_message)

        return new_messages


    @staticmethod
    def create_prompt_response_pairs_v2(messages):
        messages.reverse()

        prompt_response_pairs = []

        for i, message in enumerate(messages[:-1]):
            if (
                message["sender_name"] != "Isaiah Zettler"
                and messages[i + 1]["sender_name"] == "Isaiah Zettler"
            ):
                prompt_response_pairs.append(
                    {
                        "prompt": {
                            "sender_name": message["sender_name"],
                            "content": message["content"],
                        },
                        "response": {
                            "sender_name": messages[i + 1]["sender_name"],
                            "content": messages[i + 1]["content"],
                        },
                    }
                )

        return prompt_response_pairs

    @staticmethod
    def prepare_data_for_finetuning(prompt_response_pairs, separator_token="</s>"):
        finetuning_data = []

        for pair in prompt_response_pairs:
            prompt = pair["prompt"]["content"]
            response = pair["response"]["content"]
            finetuning_data.append(f"{prompt} {separator_token} {response}")

        return finetuning_data

    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    @staticmethod
    def clean_finetuning_data(data, output_file="cleaned_finetuning_data.txt"):
        cleaned_data = []

        for line in data:
            line = line.strip()
            if (
                line != "" and MessageProcessor.is_ascii(line) and len(line) >= 50
            ):  # Check line length
                cleaned_line = line.replace('"', "")
                cleaned_data.append(cleaned_line)

        with open(output_file, "w") as outfile:
            for line in cleaned_data:
                outfile.write(line + "\n")
