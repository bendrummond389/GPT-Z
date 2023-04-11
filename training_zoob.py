from MessageProcessor import MessageProcessor;

process_messages = MessageProcessor('./training_data_unprocessed')
process_messages.process_json_files()

