import os
import json
import codecs
from nltk.tokenize import word_tokenize

vocab = {}

def word_preprocess(text):
    text = text.replace("\n", " ")
    text = text.replace("/", " ")
    text = text.replace("\\"," ")
    text = text.replace("*", " ")
    return text


def read_single_json(file_):
    try:
        reviews = json.load(file_)
    except:
        return [],[]
    review_text = []
    score = []
    for i, review in enumerate(reviews):

        review["text"] = " ".join(word_tokenize(word_preprocess(review["text"])))

        review_words = list(map(lambda x: x.lower(),review["text"].split()))
        if len(review_words) > 200:
            continue
        #print (len(review_words))
        score.append(review["stars"])
        review_text.append(" ".join(review_words))
    file_.close()

    return review_text, score

def write_reivew_to_file(path, reviews, scores, count):
    file_path = os.path.join(path, "%06d.txt" % (count//1000))
    write_file = codecs.open(file_path, "a", "utf-8")

    for i, review in enumerate(reviews):

        dict = {"review": str(review),
                "score": str(scores[i])
                }
        string_ = json.dumps(dict)
        write_file.write(string_ + "\n")

    write_file.close()


file_path = "F:\dataset\yelp_dataset\sorted_data"

files = os.listdir(file_path)

train_size = int(len(files) * 0.7)
test_size = int(len(files) * 0.1)
valid_size = int(len(files) * 0.2)


train_filename = files[:train_size]
test_filename = files[train_size:(train_size+test_size)]
valid_filename = files[(train_size+test_size):]

review_train_number = 0
review_valid_number = 0
review_test_number = 0
score_ana_train ={}
score_ana_valid ={}
score_ana_test ={}
for i, file in enumerate(train_filename):
    file_ = codecs.open(file_path+"\\"+file,'r',"utf-8",errors="replace")
    reviews, scores = read_single_json(file_)
    review_train_number+=len(reviews)
    for score in scores:
        if score not in score_ana_train:
            score_ana_train[score] = 1
        else:
            score_ana_train[score] += 1
    write_reivew_to_file("train", reviews,scores,i)


for i, file in enumerate(valid_filename):
    file_ = codecs.open(file_path+"\\"+file,'r',"utf-8",errors="replace")
    reviews, scores = read_single_json(file_)
    write_reivew_to_file("valid", reviews,
                         scores, i)
    review_valid_number += len(reviews)
    for score in scores:
        if score not in score_ana_valid:
            score_ana_valid[score] = 1
        else:
            score_ana_valid[score] += 1

for i, file in enumerate(test_filename):
    file_ = codecs.open(file_path+"\\"+file,'r',"utf-8",errors="replace")
    reviews, scores = read_single_json(file_)

    write_reivew_to_file("test", reviews,
                         scores, i)
    review_test_number += len(reviews)
    for score in scores:
        if score not in score_ana_test:
            score_ana_test[score] = 1
        else:
            score_ana_test[score] += 1

print ("training size: %d", review_train_number)
print ("valid size: %d", review_valid_number)
print ("test size: %d", review_test_number)

print (score_ana_train)
print (score_ana_valid)
print (score_ana_test)







