import json


with open('json_history/circle-dynamics.json','r') as f:
    # Load the JSON data into a Python object
    data = json.load(f)

# Extract the initial string and the list of edits
initial_string = data['init_string']
edits = data['edits']

# Apply each edit to the string
modified_string = initial_string
print('initial_string : ',initial_string)
for edit in edits:
    loc = edit['loc']
    cut_len = edit['cut_len']
    token = edit['token']
    modified_string = modified_string[:loc] + token + modified_string[loc + cut_len:]
    print('modified_string : ',modified_string)
