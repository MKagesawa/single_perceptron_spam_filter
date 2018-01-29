import pickle

with open("train.txt") as train:
    lines_train = train.readlines()

with open("test.txt") as test:
    lines_test = test.readlines()

with open("validation.txt") as validation:
    lines_validation = validation.readlines()

with open("train+validation.txt") as t_v:
    lines_t_v = t_v.readlines()

def words(lines):
    temp = {}
    freq_list = []

    # Creating a dictionary key being the word and value being how many times it appears
    for l in lines:
        # Temporarily keep track of words in each line
        count = {}
        for w in l.split():
            if w in count:
                count[w] += 1
            else:
                count[w] = 1
        # Add words in each line only once onto the list
        for c in count:
            if c in temp:
                temp[c] += 1
            else:
                temp[c] = 1

    # Append only the words from dictionary that appear more than 25 times on different emails to the list
    for k, v in temp.items():
        if v >= 25:
            freq_list.append(k)
    return freq_list


# Pickling frequent words list generation
freq_list_object = open("frequent_words", "wb")
pickle.dump(words(lines_train), freq_list_object)
freq_list_object.close()

def feature_vector(email, freq_list):
    vector = []
    for line in email:
        # Create a list for feature vector in each line of the email file
        temp_list = []
        for word in freq_list:
            if word in line:
                temp_list.append(1)
            else:
                temp_list.append(0)
        vector.append(temp_list)
    return vector


freq_list = words(lines_train)

# Pickling train file feature vector extraction
train_vector_object = open("train_vector.pkl", "wb")
pickle.dump(feature_vector(lines_train, freq_list), train_vector_object)
train_vector_object.close()

# Pickling validation file feature vector extraction
validation_vector_object = open("validation_vector.pkl", "wb")
pickle.dump(feature_vector(lines_validation, freq_list), validation_vector_object)
validation_vector_object.close()

# Pickling test file feature vector extraction
test_vector_object = open("test_vector.pkl", "wb")
pickle.dump(feature_vector(lines_test, freq_list), test_vector_object)
test_vector_object.close()

# Pickling frequent word list
freq_word_list_object = open("freq_list.pkl", "wb")
pickle.dump(freq_list, freq_word_list_object)
freq_word_list_object.close()

# Pickling train+validation file feature vector extraction
train_validation_vector_object = open("train+validation.pkl", "wb")
pickle.dump(feature_vector(line_t, freq_list), train_validation_vector_object)
train_validation_vector_object.close()