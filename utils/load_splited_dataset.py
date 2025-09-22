import pickle


train_set_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset/train.p'
test_set_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcoco+/test.p'
train_set = pickle.load(open(train_set_path, "rb"))
test_set = pickle.load(open(test_set_path, "rb"))
sd = test_set['img2refs'][393464]
print('sd')