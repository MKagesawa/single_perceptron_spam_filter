import pickle

with open("spam_train.txt") as train:
    content = train.readlines()

with open("spam_test.txt") as test:
    test_content = test.readlines()

def split_text(content):

    # Split first 4000 lines in spam_train as train text
    a = open("train.txt", "w+")
    for l in content[:4000]:
        w = l.replace("0", "")
        w = w.replace("1", "")
        a.write(w)
    a.close()

    # Make {-1,1} label for train and pickle
    b = open("train_01.pkl", "wb")
    list_b = []
    for l in content[:4000]:
        w = l.split(" ", 1)[0] + "\n"
        w = w.replace("0", "-1")
        list_b.append(int(w))
    pickle.dump(list_b, b)
    b.close()

    # Split last 1000 lines in spam_train as validation text
    c = open("validation.txt", "w+")
    for l in content[4000:]:
        w = l.replace("0", "")
        w = w.replace("1", "")
        c.write(w)
    c.close()

    # Make {-1,1} label for validation and pickle
    d = open("validation_01.pkl", "wb")
    list_d = []
    for l in content[4000:]:
        w = l.split(" ", 1)[0] + "\n"
        w = w.replace("0", "-1")
        list_d.append(int(w))
    pickle.dump(list_d, d)
    d.close()

    # Make {-1,1} label for spam_test and pickle
    e = open("test_01.pkl", "wb")
    list_e = []
    for l in test_content:
        w = l.split(" ", 1)[0] + "\n"
        w = w.replace("0", "-1")
        list_e.append(int(w))
    pickle.dump(list_e, e)
    e.close()

    # Remove {0,1} from spam_test and make new file called test
    f = open("test.txt", "w+")
    for l in test_content[:]:
        w = l.replace("0", "")
        w = w.replace("1", "")
        f.write(w)
    f.close()

    # Remove {0,1} from spam_train and make new file called train+validation
    g = open("train+validation.txt", "w+")
    for l in content:
        w = l.replace("0", "")
        w = w.replace("1", "")
        g.write(w)
    g.close()

    # Make {-1,1} for spam_train to pickle
    h = open("train+validation_01.pkl", "wb")
    list_h = []
    for l in content:
        w = l.split(" ", 1)[0] + "\n"
        w = w.replace("0", "-1")
        list_h.append(int(w))
    pickle.dump(list_h, h)
    h.close()

split_text(content)